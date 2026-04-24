# Konfiguracja gunicorn dla BeSafeFish na Render.com
# Free tier: 512MB RAM, shared CPU
#
# gthread: pozwala obsluzyc wiele I/O-bound requestow (czekanie na DB)
# preload: laduje Flask raz, forkuje workery (oszczednosc RAM)
# timeout 120: na wypadek cold startu Render

workers = 2
threads = 2
worker_class = "gthread"
preload_app = True
timeout = 120
max_requests = 1000
max_requests_jitter = 50
accesslog = "-"
