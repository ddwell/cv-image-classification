cd image/code
#python manage.py flush
python manage.py makemigrations
python manage.py migrate --run-syncdb 
python manage.py runserver 