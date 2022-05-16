defmodule ExAF.Backend do
  @behaviour Nx.Backend

  alias Nx.Tensor, as: T

  # Creation

  def from_binary(%T{shape: shape, type: type} = out, binary, _opts) do
    shape = ExAF.to_exaf_shape(shape)
    type = ExAF.to_exaf_type(type)

    binary
    |> ExAF.Native.from_binary(shape, type)
    |> to_nx(out)
  end

  # Conversion

  def to_binary(%T{data: data}, limit, _backend_opts \\ []) do
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
end
