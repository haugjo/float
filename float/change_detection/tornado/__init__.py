"""Tornado Change Detection Methods.

This module contains implementations of popular concept drift detection methods from the tornado framework:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
Github: https://github.com/alipsgh/tornado

Copyright (C) 2021 Johannes Haug

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
