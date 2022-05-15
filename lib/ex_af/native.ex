defmodule ExAF.Native do
  use Rustler,
    otp_app: :ex_af,
    crate: "exaf_native"

  def from_binary(_, _, _), do: error()
  def to_binary(_), do: error()

  defp error, do: :erlang.nif_error(:nif_not_loaded)
end
