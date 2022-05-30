defmodule ExAF do
  @moduledoc """
  Documentation for `ExAF`.
  """

  import Nx.Shared

  @supported_types [
    {:u, 8},
    {:u, 16},
    {:u, 32},
    {:u, 64},
    {:s, 16},
    {:s, 32},
    {:s, 64},
    {:f, 16},
    {:f, 32},
    {:f, 64},
    {:c, 64},
    {:c, 128}
  ]

  def to_exaf_type(type) do
    if Enum.member?(@supported_types, type) do
      type_to_string(type)
    else
      raise ArgumentError, "ExAF does not support type: #{Nx.Type.to_string(type)}"
    end
  end

  defp type_to_string({atom, bytes}) do
    "#{atom}#{bytes}"
  end

  def to_exaf_shape(shape) when is_tuple(shape) do
    shape
    |> Tuple.to_list()
    |> to_exaf_shape
  end

  def to_exaf_shape(shape) when is_list(shape) do
    case length(shape) do
      len when len > 4 ->
        raise ArgumentError, "ExAF does not support #{len} dimensional tensors"

      len ->
        shape ++ List.duplicate(1, 4 - len)
    end
  end

  def number_to_binary(number, type) do
    match_types([type], do: <<write!(number, 0)>>)
  end
end
