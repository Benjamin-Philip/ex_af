defmodule ExAF.Backend do
  @behaviour Nx.Backend

  alias Nx.Tensor, as: T

  # Creation

  def constant(%T{type: type, shape: shape} = out, constant, backend_opts) do
    data = :binary.copy(ExAF.number_to_binary(constant, type), Nx.size(shape))
    from_binary(out, data, backend_opts)
  end

  # Conversion

  def from_binary(%T{shape: shape, type: type} = out, binary, _opts \\ []) do
    shape = ExAF.to_exaf_shape(shape)
    type = ExAF.to_exaf_type(type)

    binary
    |> ExAF.Native.from_binary(shape, type)
    |> to_nx(out)
  end

  def to_binary(%T{data: data}, limit \\ nil, _backend_opts \\ []) do
    ExAF.Native.to_binary(data, limit)
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

  defp to_nx(ref, %T{type: _type, shape: _shape} = t) do
    %{t | data: ref}
  end

  # Shape
  def reshape(%T{shape: shape} = out, %T{data: data} = tensor) do
    shape = ExAF.to_exaf_shape(shape)

    data
    |> ExAF.Native.reshape(shape)
    |> to_nx(out)
  end
end
