# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:54:02 2024

@author: mazzac3
"""

from app import create_app

application = create_app()

if __name__ == "__main__":
    application.run(debug=False)