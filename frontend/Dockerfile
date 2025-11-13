FROM nginx:alpine

# Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Web dosyaları
COPY index.html /usr/share/nginx/html/
COPY static/ /usr/share/nginx/html/static/

# Port
EXPOSE 80

# Başlat
CMD ["nginx", "-g", "daemon off;"]

