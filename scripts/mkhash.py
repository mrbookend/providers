#!/usr/bin/env python3
# scripts/mkhash.py
"""
PBKDF2-SHA256 hex hash generator for a password+salt.
Use this only if/when you re-enable an admin password gate.
"""
from __future__ import annotations
import hashlib, getpass

def pbkdf2_hex(password: str, salt: str, iterations: int = 120_000) -> str:
    data = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations)
    return data.hex()

def main():
    print("PBKDF2-SHA256 hash generator")
    salt = input("Enter SALT (16–32+ random chars): ").strip()
    pwd = getpass.getpass("Enter PASSWORD: ")
    confirm = getpass.getpass("Confirm PASSWORD: ")
    if pwd != confirm:
        print("Passwords do not match."); raise SystemExit(2)
    print("Hash (store as ADMIN_PASSWORD_HASH):")
    print(pbkdf2_hex(pwd, salt))

if __name__ == "__main__":
    main()
