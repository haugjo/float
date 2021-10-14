"""Adaptive Windowing Drift Detection Method.

Code adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Paper: Bifet, Albert, and Ricard Gavalda. "Learning from time-changing data with adaptive windowing."
Published in: Proceedings of the 2007 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2007.
URL: http://www.cs.upc.edu/~GAVALDA/papers/adwin06.pdf

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
import math
from typing import Tuple

from float.change_detection.base_change_detector import BaseChangeDetector


class Adwin(BaseChangeDetector):
    """Adwin Change Detector."""
    def __init__(self, delta: float = 0.002, reset_after_drift: bool = False):
        """Inits the change detector.

        Args:
            delta: Todo
            reset_after_drift: See description of base class.
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

        self._delta = delta
        self._adaptive_windowing = _AdaptiveWindowing(self._delta)
        self._active_change = False   # Boolean indicating whether there is an ongoing concept drift

    def reset(self):
        """Resets the change detector."""
        self._adaptive_windowing = _AdaptiveWindowing(self._delta)

    def partial_fit(self, pr: bool):
        """Updates the change detector.

        Args:
            pr: Boolean indicating a correct prediction.
                If True the prediction by the online learner was correct, False otherwise.
        """
        self._active_change = self._adaptive_windowing.set_input(pr)

    def detect_change(self) -> bool:
        """Detects global concept drift."""
        return self._active_change

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            Adwin does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone.

        Notes:
            Adwin does not raise warnings.
        """
        return False


# ----------------------------------------
# Tornado Functionality (left unchanged)
# ----------------------------------------
class _AdaptiveWindowing:
    """Adaptive windowing base class of the Tornado package.

    Notes:
        This is a class implementation from the tornado package. We changed the name from ADWIN to _AdaptiveWindowing,
        and added underscores to most attributes and functions in order to indicate that they are protected members.
        Otherwise, the code was left unchanged.
    """
    def __init__(self, delta):
        self._DELTA = delta

        self._mint_minim_longitud_window = 10
        self._mint_time = 0
        self._mint_clock = 32

        self._last_bucket_row = 0

        self._bucket_number = 0
        self._detect = 0
        self._detect_twice = 0
        self._mint_min_win_length = 5

        self._MAXBUCKETS = 5
        self._TOTAL = 0
        self._VARIANCE = 0
        self._WIDTH = 0

        self._list_row_buckets = _List()

    def _insert_element(self, value):
        self._WIDTH += 1
        self._insert_element_bucket(0, value, self._list_row_buckets.head)
        inc_variance = 0
        if self._WIDTH > 1:
            inc_variance = (self._WIDTH - 1) * (value - self._TOTAL / (self._WIDTH - 1)) * (value - self._TOTAL / (self._WIDTH - 1)) / self._WIDTH
        self._VARIANCE += inc_variance

        self._TOTAL += value
        self._compress_buckets()

    def _insert_element_bucket(self, variance, value, node):
        node.insert_bucket(value, variance)
        self._bucket_number += 1

    @staticmethod
    def _bucket_size(row):
        return int(pow(2, row))

    def _delete_element(self):
        node = self._list_row_buckets.tail
        n1 = self._bucket_size(self._last_bucket_row)
        self._WIDTH -= n1
        self._TOTAL -= node.get_total(0)
        u1 = node.get_total(0) / n1
        inc_variance = node.get_variance(0) + n1 * self._WIDTH * (u1 - self._TOTAL / self._WIDTH) * (u1 - self._TOTAL / self._WIDTH) / (n1 + self._WIDTH)
        self._VARIANCE -= inc_variance
        if self._VARIANCE < 0:
            self._VARIANCE = 0

        node.remove_bucket()
        self._bucket_number -= 1
        if node.bucket_size_row == 0:
            self._list_row_buckets.remove_from_tail()
            self._last_bucket_row -= 1
        return n1

    def _compress_buckets(self):
        cursor = self._list_row_buckets.head
        i = 0
        while True:
            k = cursor.bucket_size_row
            if k == self._MAXBUCKETS + 1:
                next_node = cursor.next
                if next_node is None:
                    self._list_row_buckets.add_to_tail()
                    next_node = cursor.next
                    self._last_bucket_row += 1
                n1 = self._bucket_size(i)
                n2 = self._bucket_size(i)
                u1 = cursor.get_total(0) / n1
                u2 = cursor.get_total(1) / n2
                inc_variance = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)
                next_node.insert_bucket(cursor.get_total(0) + cursor.get_total(1), cursor.get_variance(0) +
                                        cursor.get_variance(1) + inc_variance)
                self._bucket_number += 1
                cursor.compress_buckets_row(2)
                if next_node.bucket_size_row <= self._MAXBUCKETS:
                    break
            else:
                break
            cursor = cursor.next
            i += 1
            if cursor is None:
                break

    def set_input(self, pr):
        bln_change = False
        self._mint_time += 1
        self._insert_element(pr)

        if self._mint_time % self._mint_clock == 0 and self._WIDTH > self._mint_minim_longitud_window:
            bln_reduce_width = True
            while bln_reduce_width:
                bln_reduce_width = False
                bln_exit = False
                n0 = 0
                n1 = self._WIDTH
                u0 = 0
                u1 = self._TOTAL

                cursor = self._list_row_buckets.tail
                i = self._last_bucket_row
                while True:

                    for k in range(0, cursor.bucket_size_row):

                        n0 += self._bucket_size(i)
                        n1 -= self._bucket_size(i)
                        u0 += cursor.get_total(k)
                        u1 -= cursor.get_total(k)

                        if i == 0 and k == cursor.bucket_size_row - 1:
                            bln_exit = True
                            break

                        if n1 > self._mint_min_win_length + 1 and n0 > self._mint_min_win_length + 1 and \
                                self._bln_cut_expression(n0, n1, u0, u1):
                            self._detect = self._mint_time

                            if self._detect == 0:
                                self._detect = self._mint_time
                            elif self._detect_twice == 0:
                                self._detect_twice = self._mint_time
                            bln_reduce_width = True
                            bln_change = True
                            if self._WIDTH > 0:
                                n0 -= self._delete_element()
                                bln_exit = True
                                break
                    cursor = cursor.previous
                    i -= 1
                    if not (not bln_exit and cursor is not None):
                        break

        return bln_change

    def _bln_cut_expression(self, n0, n1, u0, u1):
        diff = math.fabs((u0 / n0) - (u1 / n1))
        n = self._WIDTH
        m = (1 / (n0 - self._mint_min_win_length + 1)) + (1 / (n1 - self._mint_min_win_length + 1))
        dd = math.log(2 * math.log(n) / self._DELTA)
        v = self._VARIANCE / self._WIDTH
        e = math.sqrt(2 * m * v * dd) + 2 / 3 * dd * m
        return diff > e


class _List:
    """List base class of the Tornado package.

    Notes:
        This is a class implementation from the Tornado package. We added underscores to most attributes and functions
        in order to indicate that they are protected members. Otherwise, the code was left unchanged.
    """
    def __init__(self):
        self._count = None
        self.head = None
        self.tail = None

        self._clear()
        self._add_to_head()

    def _is_empty(self):
        return self._count == 0

    def _clear(self):
        self.head = None
        self.tail = None
        self._count = 0

    def _add_to_head(self):
        self.head = _ListItem(self.head, None)
        if self.tail is None:
            self.tail = self.head
        self._count += 1

    def _remove_from_head(self):
        self.head = self.head.next
        if self.head is not None:
            self.head.set_previous(None)
        else:
            self.tail = None
        self._count -= 1

    def add_to_tail(self):
        self.tail = _ListItem(None, self.tail)
        if self.head is None:
            self.head = self.tail
        self._count += 1

    def remove_from_tail(self):
        self.tail = self.tail.previous
        if self.tail is None:
            self.head = None
        else:
            self.tail.set_next(None)
        self._count -= 1


class _ListItem:
    """List Item base class of the Tornado package.

    Notes:
        This is a class implementation from the Tornado package. We added underscores to most attributes and functions
        in order to indicate that they are protected members. Otherwise, the code was left unchanged.
    """
    def __init__(self, next_node=None, previous_node=None):

        self._bucket_size_row = 0
        self._MAXBUCKETS = 5

        self._bucket_total = []
        self._bucket_variance = []
        for i in range(0, self._MAXBUCKETS + 1):
            self._bucket_total.append(0)
            self._bucket_variance.append(0)

        self._next = next_node
        self.previous = previous_node
        if next_node is not None:
            next_node.previous = self
        if previous_node is not None:
            previous_node.next = self

        self._clear()

    def _clear(self):
        self._bucket_size_row = 0
        for k in range(0, self._MAXBUCKETS + 1):
            self._clear_bucket(k)

    def _clear_bucket(self, k):
        self._set_total(0, k)
        self._set_variance(0, k)

    def _insert_bucket(self, value, variance):
        k = self._bucket_size_row
        self._bucket_size_row += 1
        self._set_total(value, k)
        self._set_variance(variance, k)

    def _remove_bucket(self):
        self._compress_buckets_row(1)

    def _compress_buckets_row(self, number_items_deleted):
        for k in range(number_items_deleted, self._MAXBUCKETS + 1):
            self._bucket_total[k - number_items_deleted] = self._bucket_total[k]
            self._bucket_variance[k - number_items_deleted] = self._bucket_variance[k]
        for k in range(1, number_items_deleted + 1):
            self._clear_bucket(self._MAXBUCKETS - k + 1)
        self._bucket_size_row -= number_items_deleted

    def _set_previous(self, previous_node):
        self.previous = previous_node

    def _set_next(self, next_node):
        self._next = next_node

    def _get_total(self, k):
        return self._bucket_total[k]

    def _set_total(self, value, k):
        self._bucket_total[k] = value

    def _get_variance(self, k):
        return self._bucket_variance[k]

    def _set_variance(self, value, k):
        self._bucket_variance[k] = value
