def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


def cm_to_meter(value, power):
    """
    Convert a value in cm^power to meter^power.

    Args:
    value (float): The value to be converted.
    power (int): The power of the unit (e.g., 3 for cm^-3, 2 for cm^-2).

    Returns:
    float: The converted value.
    """
    # Determine the conversion factor based on the power
    conversion_factor = 10 ** (-2 * power)

    # Perform the conversion
    converted_value = value * conversion_factor

    return converted_value

def joule_to_ev(joules):
    """
    Convert energy from joules to electronvolts (eV).

    Args:
    joules (float): Energy in joules.

    Returns:
    float: Energy in electronvolts (eV).
    """
    # Conversion factor from joules to electronvolts
    conversion_factor = 1.602176634e-19

    # Perform the conversion
    energy_ev = joules / conversion_factor

    return energy_ev
