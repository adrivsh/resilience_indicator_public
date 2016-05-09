from fancy_round import fancy_round

def autolabel(ax,rects,color, sigdigits,  **kwargs):
    """attach labels to an existing horizontal bar plot. Passes kwargs to the text (font, color, etc)"""
    
    
    for rect in rects:
        
        #parameters of the rectangle
        h = rect.get_height()
        x = rect.get_x()
        y = rect.get_y()
        w = rect.get_width()
        
        #figures out if it is a negative or positive value
        value = x if x<0 else w

        ####
        # FORMATS LABEL
        
        #truncates the value to sigdigits digits after the coma.
        stri=str(fancy_round(value,sigdigits))
        
        #remove trailing zeros
        if "." in stri:
            while stri.endswith("0"):
                stri=stri[:-1]        
        
        #remove trailing dot
        if stri.endswith("."):
            stri=stri[:-1]        
        
        if stri=="-0":
            stri="0"
        
        #space before or after (pad)
        if value<0:
            stri = stri+' '
        else:
            stri = ' '+stri

        #actual print    
        ax.text(value, y+0.4*h, stri, ha="right" if x<0 else 'left', va='center', color=color , **kwargs)