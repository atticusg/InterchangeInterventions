import torch
from layered_intervenable_model import LayeredIntervenableModel

class SequentialLayers(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, *args):
        args = self.layers[0](*args)
        for layer in self.layers[1:]:
            args = layer(*args)
        return *args



class LIMBertLayer(torch.nn.Module):
    def __init__(self, bert,layer, final_layer_num):
        super().__init__()
        self.bert = bert
        self.layer = layer
        self.final_layer_num = final_layer_num
        self.combiner =

    def forward(self,
                hidden_states,
                layer_num=0,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True):

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.bert.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[layer_num] if head_mask is not None else None
        past_key_value = past_key_values[layer_num] if past_key_values is not None else None
        #PUSHTHIS TO GIT
        if getattr(self.bert.config, "gradient_checkpointing", False) and self.bert.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                                        create_custom_forward(layer_module),
                                        hidden_states,
                                        attention_mask,
                                        layer_head_mask,
                                        encoder_hidden_states,
                                        encoder_attention_mask,
                                        )
        else:
            layer_outputs = self.layer(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.bert.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if layer_num == self.final_layer_num:
            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        next_decoder_cache,
                        all_hidden_states,
                        all_self_attentions,
                        all_cross_attentions,
                    ]
                    if v is not None
                )
            return BaseModelOutputWithPastAndCrossAttentions(
                        last_hidden_state=hidden_states,
                        past_key_values=next_decoder_cache,
                        hidden_states=all_hidden_states,
                        attentions=all_self_attentions,
                        cross_attentions=all_cross_attentions,
                    )

        return hidden_states, \
                layer_num + 1, \
                attention_mask, \
                head_mask, \
                encoder_hidden_states, \
                encoder_attention_mask, \
                past_key_values, \
                use_cache, \
                output_attentions, \
                output_hidden_states, \
                return_dict


class LIMBERTClassifier(LayeredIntervenableModel):
    def __init__(self,
                n_classes,
                bert,
                max_length=20,
                device=None,
                debug=False):
        super().__init__(debug=debug)
        self.n_classes = n_classes
        self.bert = bert
        self.bert.train()
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        n = self.hidden_dim*max_length

        self.model_dims = [n]
        self.model_layers = torch.nn.ModuleList()
        for layer in self.bert.encoder.layer:
            self.model_layers.append(LIMBertLayer(bert, layer, len(self.bert.encoder.layer)-1))
            self.model_dims.append(n)
        self.classifier_layer = torch.nn.Linear(self.hidden_dim, self.n_classes)

        def unwrap(X):
            output = X[0]
            original_shape = copy.deepcopy(output.shape())
            output = torch.reshape(output, (original_shape[0], -1))
            rest = X[1:]
            return output, (rest, original_shape)

        def rewrap(output, stow):
            rest, original_shape = stow
            return (X[0].reshape(original_shape), *rest)


        self.build_graph(self.model_layers, self.model_dims, unwrap, rewrap)


    def forward(self, pair):
        """Computes a forward pass with input `X`."""
        X, mask = pair
        X = torch.squeeze(X)
        mask = torch.squeeze(mask)
        if self.analysis:
            self.bert.encoder = self.analysis_model
        else:
            self.bert.encoder =  self.normal_model
        output = self.bert((X, mask)).pooler_output
        output = self.classifier_layer(output)
        return output
