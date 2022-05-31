defmodule ExAF.Native do
  use Rustler,
    otp_app: :ex_af,
    crate: "exaf_native"

  # Backend management

  def backend_deallocate(_), do: error()

  # Conversion

  def from_binary(_, _, _), do: error()
  def to_binary(_, _), do: error()

  # Creation

  def iota(_, _, _), do: error()

  # Elementwise

  unary_ops =
    [:exp, :expm1, :log, :log1p, :sigmoid] ++
      [:sin, :cos, :tan, :sinh, :cosh, :tanh, :asin, :acos, :atan, :asinh, :acosh, :atanh] ++
      [:erf, :erfc] ++
      [:sqrt, :rsqrt, :cbrt] ++ [:abs, :floor, :round, :ceil, :real, :imag]

  for op <- unary_ops do
    def unquote(op)(_), do: error()
  end

  # Shape

  def reshape(_, _), do: error()

  # Type

  def as_type(_, _), do: error()

  defp error, do: :erlang.nif_error(:nif_not_loaded)
end
