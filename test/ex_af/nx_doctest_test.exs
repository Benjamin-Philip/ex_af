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

  temporarily_broken_doctests = []

  inherently_unsupported_doctests = [
    # ExAF/Arrayfire does not support s8
    from_binary: 3,
    # ExAF/Arrayfire does not support bf16
    as_type: 2,
    # ExAF/Arrayfire does not support bf16
    real: 1,
    # ExAF/Arrayfire does not support bf16
    imag: 1
  ]

  rounding_error_doctests = [
    atanh: 1,
    ceil: 1,
    cos: 1,
    cosh: 1,
    erfc: 1,
    expm1: 1,
    round: 1,
    sigmoid: 1
  ]

  os_rounding_error_doctests =
    case :os.type() do
      {:win32, _} -> [expm1: 1, erf: 1]
      _ -> []
    end

  doctest Nx,
    except:
      unimplemented_funcs
      |> Kernel.++(inherently_unsupported_doctests)
      |> Kernel.++(temporarily_broken_doctests)
      |> Kernel.++(rounding_error_doctests)
      |> Kernel.++(os_rounding_error_doctests)
      |> Kernel.++([:moduledoc])
end
