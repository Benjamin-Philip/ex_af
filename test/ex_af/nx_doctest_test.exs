defmodule ExAF.NxDoctestTest do
  @moduledoc """
  Import Nx' doctests and run them on the ExAF backend.

  Many tests are excluded for the reasons below, coverage
  for the excluded tests can be found on ExAF.NxTest.
  """

  @nx_callbacks Nx.Backend.behaviour_info(:callbacks)
  @nx_funcs Nx.__info__(:functions)

  use ExUnit.Case, async: true

  unimplemented_callbacks = @nx_callbacks -- ExAF.Backend.__info__(:functions)
  implemented_callbacks = @nx_callbacks -- unimplemented_callbacks

  # Functions broken because their callback has not been implemented.
  unimplemented_funcs =
    implemented_callbacks
    |> Keyword.keys()
    |> then(&Keyword.drop(@nx_funcs, &1))

  temporarily_broken_doctests = [
    # ExAF has not implemented reshape
    sigil_V: 2,
    # ExAF does not support c64 yet
    sigil_M: 2,
    # ExAF.Backend.constant is undefined
    to_binary: 2
  ]

  inherently_unsupported_doctests = [
    # ExAF/Arrayfire does not support s8
    from_binary: 3
  ]

  doctest Nx,
    except:
      unimplemented_funcs
      |> Kernel.++(inherently_unsupported_doctests)
      |> Kernel.++(temporarily_broken_doctests)
      |> Kernel.++([:moduledoc])
end
