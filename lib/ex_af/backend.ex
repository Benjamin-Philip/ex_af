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

  # TODO: Support a limit
  def to_binary(%T{data: data}, _limit) do
    ExAF.Native.to_binary(data)
  end

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    # TODO: Maybe add signature
    #
    # tensor
    # |> to_binary(min(limit, Nx.size(tensor)))
    # |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
    # |> maybe_add_signature(tensor)

    tensor
    |> to_binary(0)
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
  end

  defp to_nx(ref, %T{type: _type, shape: _shape} = t) do
    %{t | data: ref}
  end
end
