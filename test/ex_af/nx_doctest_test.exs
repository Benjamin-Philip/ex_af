defmodule ExAF.NxDoctestTest do
  @moduledoc """
  Import Nx' doctests and run them on the ExAF backend.

  Many tests are excluded for the reasons below, coverage
  for the excluded tests can be found on ExAF.NxTest.
  """

  use ExUnit.Case, async: true

  setup do
    Nx.default_backend(ExAF.Backend)
    :ok
  end

  doctest Nx
end
