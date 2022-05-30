defmodule ExAF.Case do
  @moduledoc """
  Test case for tensor assertions
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      import ExAF.Case
    end
  end

  # TODO: Use Nx.equal instead.

  def assert_equal(left, right) do
    b1 = Nx.to_binary(left)
    b2 = Nx.to_binary(right)

    if b1 != b2 do
      flunk("""
      Tensor assertion failed.
      left: #{inspect(left)}
      right: #{inspect(right)}
      """)
    end
  end

  # TODO: Actually test with Nx.all_close

  def assert_all_close(_left, _right, _opts \\ []) do
  end
end
