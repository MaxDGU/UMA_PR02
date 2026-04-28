from fractions import Fraction

from cognitive_tutor.algebra_uma import (
    build_correct_trace,
    build_ct_flow_trace,
    build_calibration_bin,
    format_fraction,
    generate_synthetic_equations,
    generate_synthetic_equations_matched,
    get_equation_metadata,
    HumanCalibration,
    simulate_attempt,
    simulate_attempt_v2_calibrated,
    simulate_attempt_v3_ct_flow,
    train_student,
)


def test_single_variable_equation_filter_keeps_and_drops_examples() -> None:
    expected = {
        "2x=6": ("linear_one_step", Fraction(3)),
        "x+3=7": ("linear_two_step", Fraction(4)),
        "3=y+3": ("linear_two_step", Fraction(0)),
        "y/8=9": ("linear_with_denominator", Fraction(72)),
        "3/x=6": ("reciprocal_variable_denominator", Fraction(1, 2)),
        "EG60 -3+(-2y)=7y+(-7)": ("linear_variable_both_sides", Fraction(4, 9)),
    }
    for problem, (equation_type, solution) in expected.items():
        meta = get_equation_metadata(problem)
        assert meta is not None, problem
        assert meta.equation_type == equation_type
        assert meta.solution == solution

    assert get_equation_metadata("x = 23*30") is None
    assert get_equation_metadata("EG12C x = 23*30") is None
    assert get_equation_metadata("EG2 5+4*3") is None


def test_correct_trace_and_forced_error_attempt() -> None:
    meta = get_equation_metadata("y/8=9")
    assert meta is not None
    trace = build_correct_trace(meta)
    assert trace.steps[-1] == "y = 72"
    assert "clear denominator" in trace.strategy
    assert "divide both sides" in " | ".join(trace.exec_rules)

    params = {"subjid": "0", "g": "0.01", "d": "0.1", "c": "5.0", "rt_mu": "3", "ice": "0"}
    state = train_student(params, [meta], seed=1)
    attempt = simulate_attempt(state, meta, seed=1, force_error_rule="sign error")
    row = attempt.to_row()
    assert row["correct"] == 0
    assert row["is_correct"] == 0
    assert row["answer"] == "y = -72"
    assert row["first_wrong_step"] == "y = -72"
    assert "sign error" in row["exec"]


def test_synthetic_generator_produces_solvable_single_variable_equations() -> None:
    metas = generate_synthetic_equations(40, seed=3)
    assert len(metas) == 40
    assert {meta.equation_type for meta in metas} >= {
        "linear_one_step",
        "linear_two_step",
        "linear_variable_both_sides",
        "linear_with_denominator",
        "reciprocal_variable_denominator",
    }
    assert all(format_fraction(meta.solution) for meta in metas)


def test_matched_synthetic_generator_preserves_reference_type_and_variable_distribution() -> None:
    references = [
        get_equation_metadata("2x=6"),
        get_equation_metadata("x+3=7"),
        get_equation_metadata("3=y+3"),
        get_equation_metadata("y/8=9"),
        get_equation_metadata("3/x=6"),
    ]
    assert all(meta is not None for meta in references)
    metas = generate_synthetic_equations_matched(references, count=10, seed=5)  # type: ignore[arg-type]
    assert len(metas) == 10
    assert all(get_equation_metadata(meta.equation_text) is not None for meta in metas)
    assert {meta.equation_type for meta in metas} == {meta.equation_type for meta in references if meta}
    assert {meta.variable for meta in metas} == {"x", "y"}


def test_v2_calibrated_attempt_uses_human_stuck_distribution() -> None:
    meta = get_equation_metadata("x+3=7")
    assert meta is not None
    params = {"subjid": "0", "g": "0.01", "d": "0.1", "c": "5.0", "rt_mu": "3", "ice": "0"}
    state = train_student(params, [meta], seed=1)
    calibration = HumanCalibration(
        by_equation_type={
            "linear_two_step": build_calibration_bin(
                [
                    {"correct": "0", "first_wrong_step": "x+3=7"},
                    {"correct": "0", "first_wrong_step": "x=10"},
                ]
            )
        },
        global_bin=build_calibration_bin([{"correct": "0", "first_wrong_step": "global wrong"}]),
        source_csv="toy.csv",
    )

    attempt = simulate_attempt_v2_calibrated(state, meta, calibration=calibration, seed=1)
    row = attempt.to_row()
    assert row["correct"] == 0
    assert row["answer"] in {"x+3=7", "x=10"}
    assert row["trace_step_count"] == 1
    assert "human-calibrated stuck sample" in row["exec"]


def test_v3_ct_flow_uses_problem_local_trace_and_stuck_index() -> None:
    meta = get_equation_metadata("x+3=7")
    assert meta is not None
    ct_trace = build_ct_flow_trace(meta)
    assert ct_trace.steps == ["x+3=7", "x = 4"]

    params = {"subjid": "0", "g": "0.01", "d": "0.1", "c": "5.0", "rt_mu": "3", "ice": "0"}
    state = train_student(params, [meta], seed=1)
    calibration = HumanCalibration(
        by_equation_type={
            "linear_two_step": build_calibration_bin(
                [
                    {"correct": "0", "first_wrong_step": "other-problem-step", "first_wrong_step_index": "2"},
                ]
            )
        },
        global_bin=build_calibration_bin([{"correct": "0", "first_wrong_step": "global wrong"}]),
        source_csv="toy.csv",
    )

    attempt = simulate_attempt_v3_ct_flow(state, meta, calibration=calibration, seed=1)
    row = attempt.to_row()
    assert row["correct"] == 0
    assert row["answer"] == "x = 4"
    assert row["first_wrong_step"] == "x = 4"
    assert row["trace_step_count"] == 2
    assert "other-problem-step" not in row["trace_steps_json"]
    assert "human-calibrated stuck position" in row["exec"]
