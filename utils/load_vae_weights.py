

def load_conditioned_vae_weights(model, pretrained_dict, type="encoder"):
    state = model.state_dict()

    if type == "decoder":
        state["layers.0.weight"] = pretrained_dict["decoder.layers.0.weight"]
        state["layers.0.bias"] = pretrained_dict["decoder.layers.0.bias"]
        state["layers.1.weight"] = pretrained_dict["decoder.layers.1.weight"]
        state["layers.1.bias"] = pretrained_dict["decoder.layers.1.bias"]
        state["final.weight"] = pretrained_dict["decoder.final.weight"]
        state["final.bias"] = pretrained_dict["decoder.final.bias"]

    if type == "encoder":
        state["layers.0.weight"] = pretrained_dict["encoder.layers.0.weight"]
        state["layers.0.bias"] = pretrained_dict["encoder.layers.0.bias"]
        state["layers.1.weight"] = pretrained_dict["encoder.layers.1.weight"]
        state["layers.1.bias"] = pretrained_dict["encoder.layers.1.bias"]
        state["mu.weight"] = pretrained_dict["encoder.mu.weight"]
        state["mu.bias"] = pretrained_dict["encoder.mu.bias"]
        state["log_var.weight"] = pretrained_dict["encoder.log_var.weight"]
        state["log_var.bias"] = pretrained_dict["encoder.log_var.bias"]

    return state
