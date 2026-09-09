"""
Reading the CSV a run writes.

`CsvOutput` emits one row per output step with a fixed column set -- `t`, `kGeV`
when the cutoff is known, and one column per scalar readout the model registers.

The dialect matches `DiFfRG/common/csv.hh` and `DiFfRG.file_io.csv` on the Python
side: `#` comment lines and blank lines are skipped, a cell that is not a finite
number reads back as `NaN` rather than raising, and a row whose field count does
not match the header is skipped.
"""

"""
    read_csv(path) -> Dict{String,Vector{Float64}}

Read a DiFfRG CSV output file into one column per name.

Non-numeric and non-finite entries become `NaN` rather than raising, so a run
that wrote an inf or a blank still loads and the gap is visible in the data.
"""
function read_csv(path::AbstractString)
    isfile(path) || throw(ArgumentError("no such file: $path"))
    lines = filter(!isempty, strip.(readlines(path)))
    lines = filter(line -> !startswith(line, '#'), lines)
    isempty(lines) && return Dict{String,Vector{Float64}}()

    header = String.(strip.(split(lines[1], ',')))
    cols = Dict(name => Float64[] for name in header)
    for row in lines[2:end]
        fields = split(row, ',')
        length(fields) == length(header) || continue
        for (name, field) in zip(header, fields)
            value = something(tryparse(Float64, strip(field)), NaN)
            push!(cols[name], isfinite(value) ? value : NaN)
        end
    end
    return cols
end

"""
    read_csv(result) -> Vector{Dict{String,Vector{Float64}}}

Read every CSV file a run produced.
"""
read_csv(r::RunResult) = [read_csv(f) for f in csv_files(r)]
