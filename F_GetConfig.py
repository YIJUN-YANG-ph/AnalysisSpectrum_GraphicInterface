def F_GetConfig(FileName: str = r'RRW2.8G0.5L1628.594F5mHzA2V.txt') -> dict:
    """ A function to read configuration from a given file name.

    Args:
        FileName (str): RR<width>G<gap>L<wavelength>F<freq>mHzA<amplitude>V(not important).txt
    """
    # get parameters from file name
    import re
    # if additional postfix exists, ignore it
    # FileName = re.sub(r'\(.*\)\.txt$', '.txt', FileName)
    pattern = r'RRW([0-9]+(?:\.[0-9]+)?)G([0-9]+(?:\.[0-9]+)?)L([0-9]+(?:\.[0-9]+)?)F([0-9]+(?:\.[0-9]+)?)mHzA([0-9]+(?:\.[0-9]+)?)V.*'
    match = re.match(pattern, FileName)
    # if additionl postfix exists, ignore it
    if not match:
        raise ValueError("Filename does not match the expected pattern.")
    
    config = {
        "R_width_um": float(match.group(1)),
        "gap_um": float(match.group(2)),
        "wavelength_nm": float(match.group(3)),
        "freq_mHz": float(match.group(4)),
        "amplitude_V": float(match.group(5))
    }
    return config

if __name__ == "__main__":
    FileName = r'RRW2.8G0.5L1628.594F5mHzA2V(2).txt'
    config = F_GetConfig(FileName=FileName)
    # FileName = re.sub(r'\(.*\)\.txt$', '.txt', FileName)
    print(config)