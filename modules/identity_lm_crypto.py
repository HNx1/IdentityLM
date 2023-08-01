# Implementations of some shared cryptographic functions to make the repo self-contained

from Crypto.Util import number as CryptoNumber
from math import lcm

def modular_exp(x,p,n):
    return pow(x,p,n)

def generate_keys(n_bits=32):
    # Generates and returns a public and private key using the basic RSA process

    # Get 2 n-bit primes
    p=CryptoNumber.getPrime(n_bits)
    q=CryptoNumber.getPrime(n_bits)
    while q==p:
        q=CryptoNumber.getPrime(n_bits) # check not equal

    e=2**16+1 # Standard e choice

    n=p*q # modulus for RSA

    l=lcm(p-1,q-1) # Carmichael Totient of n, special case of n=pq where p,q prime

    d=pow(e,-1,l) # inbuilt modular multiplicative inverse 

    return d,(e,n) # private key first, tuple containing both parts of public key second


def generate_keys_multi(n_bits=32,arb_size=64):
    # Generates and returns a public and private key using an adapted 3 prime RSA process
    # We'll generate 2 arb_size bit primes and 1 n_bits bit prime
    
    # Get 2 n-bit primes
    p=CryptoNumber.getPrime(arb_size)
    q=CryptoNumber.getPrime(arb_size)
    while q==p:
        q=CryptoNumber.getPrime(arb_size) # check not equal

    r=CryptoNumber.getPrime(n_bits) 
    while r<0.9*2**n_bits:
        r=CryptoNumber.getPrime(n_bits) # ensure r is close to 

    e=2**16+1 # Standard e choice

    n=p*q*r # modulus for RSA

    l=lcm(p-1,q-1,r-1) # Carmichael Totient of n, special case of n=pqr where p,q,r prime

    d=pow(e,-1,l) # inbuilt modular multiplicative inverse 

    return d,(e,n,r) # private key first, tuple containing all three parts of public key second