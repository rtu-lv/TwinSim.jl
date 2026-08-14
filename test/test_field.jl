@testset "Field2D" begin
    field = Field2D(8, 6)
    @test size(field) == (8, 6)
    @test size(field, 1) == 8
    @test eltype(field) == Float32
    @test length(field) == 48

    initialize_peak!(field, 42.0f0)
    @test center_value(field) == 42.0f0
    @test sum_state(field) == 42.0f0

    @test_throws ArgumentError Field2D(0, 4)
    @test_throws ArgumentError Field2D(4, -1)
    @test_throws BoundsError initialize_peak!(field, 1.0f0; x = 99, y = 1)

    @testset "both buffers start initialised" begin
        # Uninitialised scratch memory turns a missed cell into garbage instead
        # of an obvious zero.
        blank = Field2D(4, 4; initial = 7.0f0)
        @test all(==(7.0f0), blank.current)
        @test all(==(7.0f0), blank.next)
        @test all(iszero, Field2D(zeros(Float32, 3, 3)).next)
    end
end

@testset "swapbuffers! exchanges without copying" begin
    field = Field2D(4, 4)
    current, next = field.current, field.next
    fill!(current, 1.0f0)
    fill!(next, 2.0f0)

    swapbuffers!(field)

    # The buffers must be the same objects, in the other order: a swap that
    # copies is O(cells) per step and doubles the memory traffic of the stencil.
    @test field.current === next
    @test field.next === current
    @test all(==(2.0f0), field.current)

    swapbuffers!(field)
    @test field.current === current

    big = Field2D(512, 512)
    swapbuffers!(big)                      # warm up
    @test @allocated(swapbuffers!(big)) == 0
end

@testset "initialize_gaussian!" begin
    field = Field2D(65, 65; initial = 0.0)
    initialize_gaussian!(field; amplitude = 2.0, sigma = 4.0)
    @test center_value(field) ≈ 2.0
    @test maximum(field.current) ≈ 2.0
    @test field.current[1, 1] < 1e-6
    # Symmetric about the centre in both directions.
    @test field.current[30, 33] ≈ field.current[36, 33]
    @test field.current[33, 30] ≈ field.current[33, 36]
end
