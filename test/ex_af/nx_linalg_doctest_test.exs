defmodule ExAF.NxLinAlgDoctestTest do
  @moduledoc """
  Import Nx.LinAlg's doctests and run them on the ExAF backend.

  Many tests are excluded for the reasons below, coverage
  for the excluded tests can be found on ExAF.NxLinAlgTest.
  """

  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(ExAF.Backend)
    :ok
  end

  doctest Nx.LinAlg
end
