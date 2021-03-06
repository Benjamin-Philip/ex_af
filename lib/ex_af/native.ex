defmodule ExAF.Native do
  use Rustler,
    otp_app: :ex_af,
    crate: "exaf_native"

  alias ExAF.Helpers

  # Backend management

  def backend_deallocate(_), do: error()

  # Conversion

  def from_binary(_, _, _), do: error()
  def to_binary(_, _), do: error()

  # Creation

  def eye(_, _), do: error()
  def iota(_, _, _), do: error()

  # Elementwise

  for op <- Helpers.unary_ops() do
    def unquote(op)(_), do: error()
  end

  for op <- Helpers.binary_ops() do
    def unquote(op)(_, _), do: error()
  end

  # Shape

  def reshape(_, _), do: error()

  # Type

  def as_type(_, _), do: error()

  defp error, do: :erlang.nif_error(:nif_not_loaded)
end
