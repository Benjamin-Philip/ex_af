defmodule ExAF.Backend do
  @behaviour Nx.Backend

  alias Nx.Tensor, as: T
  alias ExAF.Native

  # Backend management

  def backend_copy(tensor, backend, opts \\ [])

  @impl true
  def backend_copy(tensor, Nx.Tensor, opts) do
    backend_copy(tensor, Nx.BinaryBackend, opts)
  end

  @impl true
  def backend_copy(tensor, backend, opts) do
    tensor
    |> Nx.to_binary()
    |> then(&backend.from_binary(tensor, &1, opts))
  end

  @impl true
  def backend_deallocate(tensor) do
    tensor
    |> from_nx
    |> Native.backend_deallocate()
  end

  @impl true
  def backend_transfer(tensor, module, opts) do
    backend_copy(tensor, module, opts)
  after
    backend_deallocate(tensor)
  end

  # Conversion

  @impl true
  def from_binary(%T{shape: shape, type: type} = out, binary, _opts \\ []) do
    shape = ExAF.to_exaf_shape(shape)
    type = ExAF.to_exaf_type(type)

    binary
    |> Native.from_binary(shape, type)
    |> to_nx(out)
  end

  @impl true
  def to_binary(tensor, limit) do
    tensor
    |> from_nx
    |> Native.to_binary(limit)
  end

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(limit)
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
    |> maybe_add_signature(tensor)
  end

  if Application.compile_env(:ex_af, :add_backend_on_inspect, true) do
    defp maybe_add_signature(result, %T{data: %{resource: ref}}) do
      '#Ref<' ++ rest = :erlang.ref_to_list(ref)

      Inspect.Algebra.concat([
        "ExAF.Backend<#{rest}",
        Inspect.Algebra.line(),
        result
      ])
    end
  else
    defp maybe_add_signature(result, _tensor) do
      result
    end
  end

  defp from_nx(%T{data: ref}) do
    ref
  end

  defp to_nx(ref, %T{type: _type, shape: _shape} = t) do
    %{t | data: ref}
  end

  # Creation

  @impl true
  def constant(%T{type: type, shape: shape} = out, constant, backend_opts) do
    data = :binary.copy(ExAF.number_to_binary(constant, type), Nx.size(shape))
    from_binary(out, data, backend_opts)
  end

  @impl true
  def iota(%T{shape: {}} = out, nil, backend_opts) do
    constant(out, 0, backend_opts)
  end

  @impl true
  def iota(out, nil, _backend_opts) do
    shape = ExAF.to_exaf_shape(out.shape)
    type = ExAF.to_exaf_type(out.type)
    tdims = ExAF.to_exaf_shape({1, 1, 1, 1})

    shape
    |> Native.iota(tdims, type)
    |> to_nx(out)
  end

  @impl true
  def iota(out, axis, _backend_opts) when axis == tuple_size(out.shape) - 1 do
    {tdims, shape} =
      out.shape
      |> Tuple.to_list()
      |> Enum.split(axis)

    type = ExAF.to_exaf_type(out.type)
    tdims = ExAF.to_exaf_shape(tdims)

    shape
    |> ExAF.to_exaf_shape()
    |> Native.iota(tdims, type)
    |> to_nx(out)
  end

  # TODO: Rewrite in an ArrayFire native manner
  # Copied from Nx.BinaryBackend

  @impl true
  def iota(%{shape: shape, type: type} = out, axis, _backend_options) do
    {dims_before, [dim | dims_after]} =
      shape
      |> Tuple.to_list()
      |> Enum.split(axis)

    # Number of repetitions of an index in memory
    repeat_blocks =
      dims_after
      |> Enum.reduce(1, &*/2)

    # Number of cycles of the counting pattern
    cycles =
      dims_before
      |> Enum.reduce(1, &*/2)

    data =
      for _ <- 1..cycles,
          i <- 0..(dim - 1),
          _ <- 1..repeat_blocks,
          into: "",
          do: ExAF.number_to_binary(i, type)

    from_binary(out, data)
  end

  # Elementwise

  # unary_ops =
  #   [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tan, :cosh, :sinh] ++
  #     [:tanh, :acos, :asin, :atan, :acosh, :asinh, :atanh, :sqrt, :rsqrt] ++
  #     [:erf, :erfc, :erf_inv, :abs, :bitwise_not, :ceil, :floor, :negate, :round, :sign] ++
  #     [:logical_not, :cbrt]

  unary_ops =
    [:exp, :expm1, :log, :log1p, :sigmoid] ++
      [:sin, :cos, :tan, :sinh, :cosh, :tanh, :asin, :acos, :atan, :asinh, :acosh, :atanh] ++
      [:erf, :erfc] ++
      [:sqrt, :rsqrt, :cbrt] ++ [:abs, :floor, :round, :ceil, :real, :imag]

  for op <- unary_ops do
    @impl true
    def unquote(op)(out, tensor) do
      type = ExAF.to_exaf_type(out.type)

      tensor
      |> from_nx()
      |> Native.unquote(op)()
      |> Native.as_type(type)
      |> to_nx(out)
    end
  end

  # Shape

  @impl true
  def reshape(out, tensor) do
    shape = ExAF.to_exaf_shape(out.shape)

    tensor
    |> from_nx()
    |> Native.reshape(shape)
    |> to_nx(out)
  end

  # Type

  @impl true
  def as_type(out, tensor) do
    type = ExAF.to_exaf_type(out.type)

    tensor
    |> from_nx()
    |> Native.as_type(type)
    |> to_nx(out)
  end
end
