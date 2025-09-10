DEVICE=$1

rsync -avz \
    --exclude=".git"\
    --exclude=".gitignore"\
    --exclude-from=".gitignore"\
    ./ agranados-24@$DEVICE:/home/infres/agranados-24/meshgraphnets

# agranados-24@gpu9.enst.fr:/home/infres/agranados-24/meshgraphnets