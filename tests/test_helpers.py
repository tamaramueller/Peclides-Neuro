from __future__ import division
import pytest
import helpers


def test_get_specificity():
    spec = helpers.get_specificity(reslist=[1,0,0], truevals=[1,0,0])
    print(spec)
    assert spec == 1


def test_get_sensitivity():
    sens = helpers.get_sensitivity(reslist=[1,1,0], truevals=[1,1,1])
    print(sens)
    assert sens == 2/3

