"""
Web application module
"""

from flask import Flask

def create_app():
    """Flask uygulamasını oluştur"""
    from .routes import app
    return app

