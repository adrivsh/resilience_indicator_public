import warnings


def replace_with_warning(series_in,dico,ignore_case=True, joiner=", "):
        
    # preprocessing    
    series_to_use = series_in
    dico_to_use= dico
    
    if ignore_case:
        series_to_use =series_to_use.str.lower()
        dico_to_use.index=dico_to_use.index.str.lower()
     
    #processing    
    out=series_to_use.replace(dico_to_use)
    
    # post processing
    are_missing = ~out.isin(dico_to_use)
    
    if are_missing.sum()>0:
        warnings.warn("These entries were not found in the dictionary: "+joiner.join(series_in[are_missing].unique()))

    return out