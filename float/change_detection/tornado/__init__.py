"""Tornado Change Detection Methods.

This module contains implementations of popular concept drift detection methods from the tornado framework:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
Github: https://github.com/alipsgh/tornado

Copyright (C) 2022 Johannes Haug.
"""
from .adwin import Adwin
from .cusum import Cusum
from .ddm import DDM
from .eddm import EDDM
from .ewma import EWMA
from .fhddm import FHDDM
from .fhddms import FHDDMS
from .fhddms_add import FHDDMSAdd
from .hddm_a import HDDMA
from .hddm_w import HDDMW
from .mddm_a import MDDMA
from .mddm_e import MDDME
from .mddm_g import MDDMG
from .page_hinkley import PageHinkley
from .rddm import RDDM
from .seqdrift2 import SeqDrift2

__all__ = ['Adwin', 'Cusum', 'DDM', 'EDDM', 'EWMA', 'FHDDM', 'FHDDMS', 'FHDDMSAdd', 'HDDMA', 'HDDMW', 'MDDMA', 'MDDME',
           'MDDMG', 'PageHinkley', 'RDDM', 'SeqDrift2']
