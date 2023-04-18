import torch
from torch import nn

from transformers import GPT2Model, GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Block,GPT2Attention, GPT2MLP
from transformers.modeling_utils import Conv1D
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from .layer import final_hidden

class GPT2ForSequenceClassificationCont(GPT2ForSequenceClassification):

    def __init__(self, config):
        super(GPT2ForSequenceClassification,self).__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2ModelCont(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


class GPT2ModelCont(GPT2Model):
    def __init__(self, config):
        super(GPT2Model,self).__init__(config)

        ## Divide hidden size by 2 so input_embedding and
        ## position_embedding will have same size
        if config.hidden_size%2 != 0:
            raise ValueError('Hidden size must be specified')
        self.wte_mode   = config.wte_mode
        self.vocab_size = config.vocab_size
        self.output     = config.output
        self.embed_method = config.embed_method # concat,add
        self.hidden_size = config.hidden_size

        if self.embed_method == 'concat':
            self.embed_dim  = int(config.hidden_size/2)
        elif self.embed_method == 'add':
            self.embed_dim  = int(config.hidden_size)
        else:
            raise ValueError('Input valid embed_method: {}',format(self.embed_method))

        if self.wte_mode == 'identity':
            if self.vocab_size != self.embed_dim:
                raise ValueError('identity for wte_mode cannot be used as vocab_size and embed_dim do not match.')
            self.wte = nn.Identity()
        elif self.wte_mode == 'linear':
            self.wte = nn.Linear(config.vocab_size, self.embed_dim)
        elif self.wte_mode == 'custom':
            raise Exception('custom not available yet')
        elif self.wte_mode == 'Embedding':
            self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        else:
            raise ValueError('Input valid wte_mode [identity, linear, Embedding]')

        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2BlockCont(config) for _ in range(config.num_hidden_layers)])

        ## Replace layernorm with other operations suitable for output type
        # self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.ln_f = final_hidden(self.hidden_size, self.vocab_size)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        generate=False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            ## Modified to allow flexibility for wte operations
            input_shape = input_ids.size()
            if self.wte_mode == 'Embedding':
                # input_ids (n_batch, seq_len) is index for look up table
                input_ids = input_ids.view(-1, input_shape[-1])
            else:
                # input_ids (n_batch, seq_len, n_embd) is vector
                input_shape = input_shape[:-1]
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        # Modified
        # changed from addition of embeddings to concatentation
        # hidden_states = inputs_embeds + position_embeds
        position_embeds = position_embeds.repeat((len(inputs_embeds),1,1))
        hidden_states = torch.cat((inputs_embeds,position_embeds),axis=-1)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        if generate and self.output == 'binary':
            # hidden_states = torch.round(hidden_states)
            m =  torch.distributions.Bernoulli(hidden_states)
            hidden_states = m.sample()

        ## comment out
        # hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



class GPT2BlockCont(GPT2Block):

    def __init__(self,config):
        # init with grandparent class (nn.module)
        super(GPT2Block,self).__init__()

        # Reimplement init with modification
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2AttentionCont(config) # Changed from GPT2Attention
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2AttentionCont(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)


class GPT2AttentionCont(GPT2Attention):

    def __init__(self, config, is_cross_attention=False):

        # Temporarily set default num_heads so init will go through
        # Potentially change to init with grandparent class so no duplicates are created
        num_heads=int(config.n_head)
        config.n_head = 1

        super().__init__(config, is_cross_attention)

        # Overwrite init for the following
        config.n_head = num_heads
        self.num_heads = num_heads
        self.head_dim = config.head_dim

        # Create
        self.n_total = self.head_dim * self.num_heads
        self.split_size = self.n_total

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.n_total, self.embed_dim)
            self.q_attn = Conv1D(self.n_total, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.n_total, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.n_total)
