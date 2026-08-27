from math import isclose
from cl.common.information_validity import (
    acquisition_mixture, aggregate_pattern_metrics, bayes_accuracy, cross_entropy,
    deterministic_accuracy, enumerated_iid_majority_accuracy, iid_candidate_coverage, iid_majority_accuracy,
    independent_trajectory_success, information_validity, sampling_accuracy,
)

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

def test_uniform_four_class_null_accuracy():
    uniform=(.25,)*4
    assert sampling_accuracy(uniform,uniform)==.25

def test_known_nonuniform_bayes_distribution_and_proper_score():
    truth=(.50,.25,.15,.10); machine=(.45,.25,.20,.10); null=(.25,)*4
    assert bayes_accuracy(truth)==deterministic_accuracy(truth,0)==.50
    assert isclose(sampling_accuracy(truth,null),.25)
    assert isclose(sampling_accuracy(truth,machine),.3275)
    assert isclose(cross_entropy(truth,null)-cross_entropy(truth,machine),.16879272656651256)

def test_positive_context_information_can_leave_residual_entropy():
    rows=[("x","a",0,"r")]*3+[("x","a",1,"r")]+[("x","b",0,"r")]+[("x","b",1,"r")]*3
    report=information_validity(rows,"r")
    assert report.conditional_mutual_information>0
    assert report.h_y_given_xd>0
    assert report.bayes_ceiling==.75

def test_iid_majority_formula_matches_exact_enumeration():
    assert isclose(iid_majority_accuracy(.6,5),enumerated_iid_majority_accuracy(.6,5))

def test_candidate_coverage_is_not_majority_accuracy():
    assert isclose(iid_candidate_coverage(.4,3),.784)
    assert isclose(iid_majority_accuracy(.4,3),.352)

def test_correlated_repetitions_defeat_iid_gain():
    majority,coverage=aggregate_pattern_metrics({(True,True,True):.6,(False,False,False):.4})
    assert majority==coverage==.6
    assert iid_majority_accuracy(.6,3)>.6
