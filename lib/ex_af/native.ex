defmodule ExAF.Native do
  use Rustler,
    otp_app: :ex_af,
    crate: "exaf_native"

  # Conversion

  def from_binary(_, _, _), do: error()
  def to_binary(_, _), do: error()

  # Creation

  def iota(_, _, _), do: error()

  # Shape

  def reshape(_, _), do: error()

  # Type

  def as_type(_, _), do: error()

  defp error, do: :erlang.nif_error(:nif_not_loaded)
end
