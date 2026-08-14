# Turning residuals into a decision, and scoring that decision honestly.
#
# The bookkeeping here is deliberately not the exercise. Choosing a threshold —
# deciding which of the two errors hurts the stakeholder more — is the exercise,
# and it is not a computational question. What this file removes is the fiddly
# counting that stands between a student and that decision.

"""
    flag_exceedances(residuals, threshold) -> Vector{Bool}

Flag samples whose residual exceeds `threshold` in magnitude. The simplest
possible detector, and the baseline everything else is compared against.
"""
flag_exceedances(residuals, threshold) = [abs(r) > threshold for r in residuals]

"""
    DetectionReport

The result of scoring flags against ground truth.

- `delay` — samples between the first anomalous sample and the first flag raised
  during it. `missing` when the anomaly was never detected.
- `false_positive_rate` — flags raised on normal samples, over the number of
  normal samples. The quantity a stakeholder feels as "how often does it cry
  wolf".
- `recall` — anomalous samples flagged, over anomalous samples.
- `precision` — flags that were right, over flags raised.

Precision and recall are `NaN` when their denominator is zero, which is
information rather than an error: no flags raised at all, or no anomaly present.
"""
struct DetectionReport
    detected::Bool
    delay::Union{Int,Missing}
    true_positives::Int
    false_positives::Int
    false_negatives::Int
    true_negatives::Int
    false_positive_rate::Float64
    precision::Float64
    recall::Float64
    threshold::Float64
end

"""
    detection_report(flags, truth; threshold = NaN) -> DetectionReport

Score a boolean flag series against boolean ground truth.

The detection delay is measured from the first `true` in `truth` to the first
`true` in `flags` at or after it — so a detector that fires *before* the anomaly
begins is scoring a false positive, not an early detection.
"""
function detection_report(flags::AbstractVector{Bool}, truth::AbstractVector{Bool};
                          threshold::Real = NaN)
    length(flags) == length(truth) || throw(DimensionMismatch(
        "flags and truth must have equal length, got $(length(flags)) and $(length(truth))"))

    true_positives = count(flags .& truth)
    false_positives = count(flags .& .!truth)
    false_negatives = count(.!flags .& truth)
    true_negatives = count(.!flags .& .!truth)

    normal = false_positives + true_negatives
    false_positive_rate = normal == 0 ? NaN : false_positives / normal
    precision = (true_positives + false_positives) == 0 ? NaN :
                true_positives / (true_positives + false_positives)
    recall = (true_positives + false_negatives) == 0 ? NaN :
             true_positives / (true_positives + false_negatives)

    onset = findfirst(truth)
    delay = missing
    detected = false
    if onset !== nothing
        hit = findnext(flags, onset)
        if hit !== nothing && truth[hit]
            delay = hit - onset
            detected = true
        end
    end

    return DetectionReport(detected, delay, true_positives, false_positives,
                           false_negatives, true_negatives, false_positive_rate,
                           precision, recall, Float64(threshold))
end

"""
    detection_report(residuals, truth, threshold) -> DetectionReport

Flag by magnitude and score in one step.
"""
detection_report(residuals::AbstractVector{<:Real}, truth::AbstractVector{Bool},
                 threshold::Real) =
    detection_report(flag_exceedances(residuals, threshold), truth; threshold)

function Base.show(io::IO, ::MIME"text/plain", report::DetectionReport)
    println(io, "DetectionReport")
    isnan(report.threshold) || @printf(io, "  threshold            %.4g\n", report.threshold)
    @printf(io, "  detected             %s\n", report.detected ? "yes" : "no")
    @printf(io, "  delay                %s samples\n",
            report.delay === missing ? "never" : string(report.delay))
    @printf(io, "  true positives       %d\n", report.true_positives)
    @printf(io, "  false positives      %d\n", report.false_positives)
    @printf(io, "  false negatives      %d\n", report.false_negatives)
    @printf(io, "  false positive rate  %.4f\n", report.false_positive_rate)
    @printf(io, "  precision            %.4f\n", report.precision)
    @printf(io, "  recall               %.4f", report.recall)
    return nothing
end

function Base.show(io::IO, report::DetectionReport)
    @printf(io, "DetectionReport(delay=%s, FP=%d, FPR=%.3f)",
            report.delay === missing ? "never" : string(report.delay),
            report.false_positives, report.false_positive_rate)
    return nothing
end

"""
    threshold_sweep(residuals, truth, thresholds) -> Vector{DetectionReport}

Score the same residuals at several thresholds, so the trade-off can be read off
rather than argued about.

```julia
for report in threshold_sweep(residuals, observed.anomalous, [0.5, 1.0, 2.0, 4.0])
    println(report)
end
```

There is no best row. A lower threshold detects sooner and cries wolf more
often; the choice depends on what a false alarm costs the operator against what
a missed fault costs. Lab Work 4 asks for the choice to be justified in those
terms, not in terms of a score.
"""
threshold_sweep(residuals, truth, thresholds) =
    [detection_report(residuals, truth, threshold) for threshold in thresholds]
