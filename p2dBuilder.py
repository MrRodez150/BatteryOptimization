
from EqnElectrode import ElectrodeEquation
from EqnSeparator import SeparatorEquation
from EqnCurrentCollector import CurrentCollectorEquation


def get_battery_sections(p_dat, n_dat, o_dat, a_dat, z_dat, Iapp):
    peq = ElectrodeEquation(p_dat, o_dat, a_dat, z_dat)
    neq = ElectrodeEquation(n_dat, o_dat, a_dat, z_dat)
    
    sepq = SeparatorEquation(o_dat, p_dat, n_dat)
    accq = CurrentCollectorEquation(a_dat, Iapp)
    zccq = CurrentCollectorEquation(z_dat, Iapp)

    return peq, neq, sepq, accq, zccq
