from math import isclose
from cl.common.information_validity import acquisition_mixture, independent_trajectory_success, information_validity

def test_context_resolves_uniform_three_way_target():
    rows=[]
    for y in range(3): rows.append(("query",f"context-{y}",y,"uniform-permutation"))
    report=information_validity(rows,"uniform-permutation",6,2)
    assert isclose(report.h_y_given_x,1.584962500721156)
    assert report.h_y_given_xd==0
    assert isclose(report.conditional_mutual_information,report.h_y_given_x)
    assert report.bayes_ceiling==1

def test_ambiguous_context_has_subunit_bayes_ceiling():
    rows=[("x","same",0,"r"),("x","same",1,"r")]
    report=information_validity(rows,"r")
    assert report.h_y_given_xd==1 and report.bayes_ceiling==.5

def test_error_and_acquisition_derivations():
    assert isclose(independent_trajectory_success(.1,3),.729)
    assert isclose(acquisition_mixture(.25,.9,.1),.3)
