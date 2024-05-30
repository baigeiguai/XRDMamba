
def add_scalars(writer,epoch_idx,key_values):
    for key,value in key_values.items():
        writer.add_scalar(key,value,epoch_idx)