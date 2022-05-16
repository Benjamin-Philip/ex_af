defmodule ExAF do
  @moduledoc """
  Documentation for `ExAF`.
  """

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
    {:f, 64}
  ]

  def to_exaf_type(type) do
    if Enum.member?(@supported_types, type) do
      type_to_string(type)
    else
      raise ArgumentError, "ExAF does not support type: #{inspect(type)}"
    end
  end

  defp type_to_string({atom, bytes}) do
    "#{atom}#{bytes}"
  end

  def to_exaf_shape(shape) do
    shape = Tuple.to_list(shape)

    case length(shape) do
      len when len > 4 ->
        raise ArgumentError, "ExAF does not support #{len} dimensional tensors"

      len ->
        shape ++ List.duplicate(1, 4 - len)
    end
  end
end
