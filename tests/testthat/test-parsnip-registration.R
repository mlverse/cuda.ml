test_that("loading cuda.ml does not load optional parsnip", {
  loaded <- callr::r(function() {
    loadNamespace("cuda.ml")
    isNamespaceLoaded("parsnip")
  })

  expect_false(loaded)
})

test_that("parsnip registration works in either namespace load order", {
  skip_if_not_installed("parsnip")

  orders <- list(
    c("cuda.ml", "parsnip"),
    c("parsnip", "cuda.ml")
  )

  for (order in orders) {
    registered <- callr::r(
      function(order) {
        for (package in order) {
          loadNamespace(package)
        }

        specification <- parsnip::set_engine(
          parsnip::linear_reg(),
          "cuda.ml"
        )
        parsnip::translate(specification)
        TRUE
      },
      args = list(order = order)
    )

    expect_true(registered, info = paste(order, collapse = " then "))
  }
})
