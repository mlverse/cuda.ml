

# read an MNIST file (encoded in IDX format)
read_idx <- function(file) {
  conn <- gzfile(file, open = "rb")
  on.exit(close(conn))

  # read the magic number as sequence of 4 bytes
  magic <- readBin(conn, what = "raw", n = 4, endian = "big")

  stopifnot(
    magic[1:2] == 0,  # check magic number starts with 2 0 bytes)
    magic[3] == 0x08) # assert data format is 8-bit unsigned ints)

  ndims <- as.integer(magic[[4]])

  # read the dimensions (32-bit integers)
  dims <- readBin(conn, what = "integer", n = ndims, endian = "big")

  # read the rest in as 8 bit unsigned ints
  data <- readBin(conn, what = "integer", size = 1, signed = FALSE,
                  n = prod(dims), endian = "big")

  array(data, dim = rev(dims))
}




fetch <- function(url, data_dir = "data-raw") {
  if(!dir.exists(data_dir))
    dir.create(data_dir)

  path <- file.path(data_dir, basename(url))
  if(!file.exists(path))
    download.file(url, destfile = path)

  path
}

# For information about MNIST, see http://yann.lecun.com/exdb/mnist/

mnist_images <-
  "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz" |>
  fetch() |> read_idx()

mnist_labels <-
  "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz" |>
  fetch() |> read_idx()


plot_mnist <- function(idxs) {
  opar <- par(mfrow = n2mfrow(length(idxs)),
              mar = c(0, 0, 1.5, 0), oma = c(0, 0, 3, 0))
  on.exit(par(opar))
  for (i in idxs) {
    img <- mnist_images[,,i]
    img <- t(apply(img, 1, rev)) # flip and rotate
    image(img, main = mnist_labels[i],
          col = gray(seq(1, 0, length.out = 255)),
          useRaster = TRUE, axes = FALSE, asp = 1, frame.plot = FALSE)
    rect(0, 0, 1, 1)
  }
  title("MNIST", outer = TRUE)
}

## Confirm MNIST loaded properly
# plot_mnist(1:64)
