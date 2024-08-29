import jax
import jax.numpy as jnp
import imageio

WIDTH, HEIGHT = 6000, 6000


def get_primes(n):
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Returns  a list of primes < n """
    sieve = [True] * (n // 2)
    for i in range(3, int(n ** 0.5) + 1, 2):
        if sieve[i // 2]:
            sieve[i * i // 2::i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
    return [2] + [2 * i + 1 for i in range(1, n // 2) if sieve[i]]


def smooth(x):
    z = jnp.linspace(-5., 5., 11)
    kernel = jnp.exp(- (z[None, :] ** 2 + z[:, None] ** 2))
    return jax.scipy.signal.convolve(x, kernel, mode='same')


def colormap(x):
    background = jnp.array([18, 18, 18])
    point = jnp.array([242, 245, 255])

    x = jnp.clip(x, 0, 1)
    x = (1 - x[..., None]) * background + x[..., None] * point
    return jnp.clip(x, 0, 255).astype(jnp.uint8)


if __name__ == '__main__':
    spiral_width = 1.5
    max_distance = jnp.sqrt(HEIGHT ** 2 + WIDTH ** 2) / 2 * 1.1
    n = int((max_distance / spiral_width) ** 2)
    primes = jnp.array(get_primes(n))
    print(f'Found {len(primes)} prime numbers smaller than {n}.')

    prime_sieve = jnp.zeros(n)
    prime_sieve = prime_sieve.at[primes].set(1.)

    theta = jnp.sqrt(jnp.arange(n)) * 2 * jnp.pi
    x = spiral_width * theta[:, None] / (2 * jnp.pi) * jnp.stack([- jnp.sin(theta), jnp.cos(theta)], axis=-1)

    canvas, _, _ = jnp.histogram2d(x[:, 0], x[:, 1], bins=(HEIGHT, WIDTH), range=[[- HEIGHT / 2, HEIGHT / 2], [- WIDTH / 2, WIDTH / 2]], weights=prime_sieve+.1)
    canvas = smooth(canvas.astype(jnp.float32))
    canvas = colormap(canvas)

    imageio.imwrite('prime.png', canvas)
