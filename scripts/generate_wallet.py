from mnemonic import Mnemonic
import hashlib

m = Mnemonic("english")
wordlist = m.wordlist

# "apple" is index 75 in BIP-39 wordlist
# 23 words of "apple" = 23 × 11 bits = 253 bits
# We need to find a 24th word (11 bits) such that
# the last 8 bits match the SHA-256 checksum

base = "winter " * 23

for word in wordlist:
    phrase = base + word
    if m.check(phrase):
        print(f"Valid mnemonic found: {phrase}")
        break