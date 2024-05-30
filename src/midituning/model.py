import torch
from transformers import PreTrainedModel, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import (
    LoraConfig,
    PeftConfig,
    LoraModel,
    PeftModelForCausalLM,
    get_peft_model_state_dict,
)
from peft.utils import WEIGHTS_NAME
from accelerate.hooks import remove_hook_from_submodules
import os
from typing import Tuple, Optional, Union, List, Any
from data_utils import rank0_print

ADAPTER_USER = "adapter_user"
ADAPTER_SYSTEM = "adapter_system"


class DialogModel(PeftModelForCausalLM):
    """
    DialogModel is wrapped from the PeftModelForCausalLM class.
    """
    def __init__(self, model: PreTrainedModel, peft_config: LoraConfig, **kwargs: Any):
        # Initialize the model with the system adapter
        super().__init__(model, 
                         peft_config, 
                         adapter_name=ADAPTER_SYSTEM)
        peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
        
        user_lora_config = LoraConfig(
            r=peft_config.r,
            lora_alpha=peft_config.lora_alpha,
            target_modules=["q_proj"],
            lora_dropout=peft_config.lora_dropout,
            bias=peft_config.bias,
            task_type="CAUSAL_LM",
        )

        # Wrap the model with the user adapter
        self.base_model = LoraModel(
            self.base_model, 
            user_lora_config, 
            adapter_name=ADAPTER_USER
        )
        self.add_adapter(
            adapter_name=ADAPTER_USER,
            peft_config=user_lora_config
        )

        self.weight_beta = kwargs.get("weight_beta", 0.0)

        self.config = getattr(self.base_model, "config", {"model_type": "custom"})
        # Set to enable caching
        self.config.use_cache = True

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1
    

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return super().state_dict()
    
    def load_state_dict(self, state_dict, strict: bool = True):
        load_result = super().load_state_dict(state_dict, strict=strict)
        # Combine the missing keys and unexpected keys.
        missing_keys, unexpected_keys = [], []
        if len(load_result.missing_keys) != 0:
            missing_keys.extend(load_result.missing_keys)
        if len(load_result.unexpected_keys) != 0:
            unexpected_keys.extend(load_result.unexpected_keys)
        # Return the same thing as PyTorch load_state_dict function.
        return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)
    
    def load_adapter(self, model_id: str, adapter_name: str = "default", is_trainable: bool = False, **kwargs: Any):
        if adapter_name == "default":
            user_model_path = os.path.join(model_id, ADAPTER_USER)
            super().load_adapter(
                user_model_path,
                adapter_name=ADAPTER_USER,
                is_trainable=is_trainable,
                **kwargs
            )
            system_model_path = os.path.join(model_id, ADAPTER_SYSTEM)
            super().load_adapter(
                system_model_path,
                adapter_name=ADAPTER_SYSTEM,
                is_trainable=is_trainable,
                **kwargs
            )
        else:
            super().load_adapter(
                model_id,
                adapter_name=adapter_name,
                is_trainable=is_trainable,
                **kwargs
            )
    
    def get_base_model(self):
        return self.base_model.model.model
    
    def forward(self,
        instruct_ids: torch.LongTensor = None,
        role_ids: torch.LongTensor = None,
        context_ids: torch.LongTensor = None,
        target_ids: torch.LongTensor = None,
        instruct_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the model.
        """

        # Set active adapter
        self.set_adapter(ADAPTER_SYSTEM)
        position_ids = instruct_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(instruct_mask == 0, 1)

        output = self.get_base_model()(
            input_ids=instruct_ids,
            attention_mask=instruct_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True
        )
        past_key_values = output.past_key_values

        total_loss = None
        logits = None
        user_attention_mask = torch.zeros(
            instruct_mask.shape, dtype=torch.bool, device=instruct_mask.device
        )
        past_attention_mask = instruct_mask

        total_turns = context_ids.shape[1]
        seq_length = context_ids.shape[2]
        for turn_num in range(total_turns):
            #print("turn_num: ", turn_num)
            user_attention_mask = torch.cat([user_attention_mask, attention_mask[:, turn_num, :]], dim=1)
            past_attention_mask = torch.cat([past_attention_mask, attention_mask[:, turn_num, :]], dim=1)

            if role_ids[:, turn_num].sum() == 0:
                #print("activate user adapter...")
                # Set active adapter
                self.set_adapter(ADAPTER_USER)

                # Note: we should set postion_ids explicitly due to padding tokens among turns.
                user_position_ids = user_attention_mask.long().cumsum(-1) - 1
                user_position_ids.masked_fill_(user_attention_mask == 0, 1)
                user_position_ids = user_position_ids[:, -seq_length:].unsqueeze(-1)

                output = self.get_base_model()(
                    input_ids=context_ids[:, turn_num, :],
                    attention_mask=user_attention_mask,
                    past_key_values=past_key_values,
                    position_ids=user_position_ids,
                    labels=target_ids[:, turn_num, :] if target_ids is not None else None,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = output.past_key_values

                if output.loss is not None:
                    if total_loss is None:
                        total_loss = self.weight_beta * output.loss
                    else:
                        total_loss += self.weight_beta * output.loss
            else:
                #print("activate system adapter...")
                # Set active adapter
                self.set_adapter(ADAPTER_SYSTEM)   

                # Note: we should set postion_ids explicitly due to padding tokens among turns.
                position_ids = past_attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(past_attention_mask == 0, 1)
                position_ids = position_ids[:, -seq_length:].unsqueeze(-1)

                output = self.get_base_model()(
                    input_ids=context_ids[:, turn_num, :],
                    attention_mask=past_attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    labels=target_ids[:, turn_num, :] if target_ids is not None else None,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = output.past_key_values

                if output.loss is not None:
                    if total_loss is None:
                        total_loss = output.loss
                    else:
                        total_loss += output.loss
        if total_loss is not None:
            # Average the loss
            total_loss = total_loss / total_turns
            
        if not return_dict:
            output = (logits, past_key_values)
            return (total_loss,) + output if total_loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=past_key_values,
        )
    

    def generate(self, generation_config: GenerationConfig, **kwargs):
        """
        Generates sequences of token ids for models with a language modeling head.
        Args:
            generation_config (`~generation.GenerationConfig`):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            kwargs (`Dict[str, Any]`):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Returns:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`
        """
        instruct_ids = kwargs.get("instruct_ids")
        role_ids = kwargs.get("role_ids")
        context_ids = kwargs.get("context_ids")
        instruct_mask = kwargs.get("instruct_mask")
        attention_mask = kwargs.get("attention_mask")

        # Set active adapter
        self.set_adapter(ADAPTER_SYSTEM)
        position_ids = instruct_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(instruct_mask == 0, 1)

        output = self.get_base_model()(
            input_ids=instruct_ids,
            attention_mask=instruct_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True
        )
        past_key_values = output.past_key_values

        user_attention_mask = torch.zeros(
           instruct_mask.shape, 
            dtype=torch.bool, 
            device=instruct_mask.device
        )
        past_attention_mask = instruct_mask

        total_turns = context_ids.shape[1]
        seq_length = context_ids.shape[2]
        outputs = None
        for turn_num in range(total_turns):
            if turn_num  == total_turns - 1:
                # Set active adapter
                self.set_adapter(ADAPTER_SYSTEM)

                input_ids = context_ids[:, turn_num, :][:, -1:]
                attention_mask = attention_mask[:, turn_num, :][:, -1:]
                attention_mask = torch.cat([past_attention_mask, attention_mask], dim=1)
                
                # Create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1)
                
                # Perform generation
                outputs = self.get_base_model().generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                    generation_config=generation_config
                )
            else:
                user_attention_mask = torch.cat([user_attention_mask, attention_mask[:, turn_num, :]], dim=1)
                past_attention_mask = torch.cat([past_attention_mask, attention_mask[:, turn_num, :]], dim=1)

                if role_ids[:, turn_num].sum() == 0:
                    #print("turn: {} user adapter...".format(turn_num))
                    # Set active adapter
                    self.set_adapter(ADAPTER_USER)
                    
                    # Note: we should set postion_ids explicitly due to padding tokens among turns.
                    user_position_ids = user_attention_mask.long().cumsum(-1) - 1
                    user_position_ids.masked_fill_(user_attention_mask == 0, 1)
                    user_position_ids = user_position_ids[:, -seq_length:].unsqueeze(-1)

                    output = self.get_base_model()(
                        input_ids=context_ids[:, turn_num, :],
                        attention_mask=user_attention_mask,
                        past_key_values=past_key_values,
                        position_ids=user_position_ids,
                        use_cache=True,
                        return_dict=True
                    )
                    past_key_values = output.past_key_values
                else:
                    #print("turn: {} system adapter...".format(turn_num))
                    # Set active adapter
                    self.set_adapter(ADAPTER_SYSTEM)
                    
                    # Note: we should set postion_ids explicitly due to padding tokens among turns.
                    position_ids = past_attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(past_attention_mask == 0, 1)
                    position_ids = position_ids[:, -seq_length:].unsqueeze(-1)

                    output = self.get_base_model()(
                        input_ids=context_ids[:, turn_num, :],
                        attention_mask=past_attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        use_cache=True,
                        return_dict=True
                    )
                    past_key_values = output.past_key_values
        
        return outputs

    
    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
        selected_adapters: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        if safe_serialization:
            rank0_print("Safe serialization is ignored.")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        os.makedirs(save_directory, exist_ok=True)
        self.create_or_update_model_card(save_directory)

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if peft_config.is_prompt_learning
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            if peft_config.task_type is None:
                # deal with auto mapping
                base_model_class = self._get_base_model_class(
                    is_prompt_tuning=peft_config.is_prompt_learning,
                )
                parent_library = base_model_class.__module__

                auto_mapping_dict = {
                    "base_model_class": base_model_class.__name__,
                    "parent_library": parent_library,
                }
            else:
                auto_mapping_dict = None

            peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode
    
    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs: Any,
    ):
        r"""
        Instantiate a PEFT model from a pretrained model and loaded PEFT weights.

        Note that the passed `model` may be modified inplace.

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to be adapted. The model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
            model_id (`str` or `os.PathLike`):
                The name of the PEFT configuration to use. Can be either:
                    - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                      method (`./my_peft_config_directory/`).
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to be loaded. This is useful for loading multiple adapters.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and use for
                inference
            config ([`~peft.PeftConfig`], *optional*):
                The configuration object to use instead of an automatically loaded configuation. This configuration
                object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
                loaded before calling `from_pretrained`.
            kwargs: (`optional`):
                Additional keyword arguments passed along to the specific PEFT configuration class.
        """
        # load the config
        if config is None:
            config_path = os.path.join(model_id, ADAPTER_SYSTEM)
            config = LoraConfig.from_pretrained(config_path, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable
        
        model = cls(model, config)

        # load the weights
        model.load_adapter(
            model_id, 
            adapter_name=adapter_name, 
            is_trainable=is_trainable, 
            **kwargs
        )
        
        return model
