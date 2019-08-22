# Script to download processed annotations
wget --header="Host: doc-14-5s-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.87 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3" --header="Accept-Language: en-US,en;q=0.9" --header="Referer: https://drive.google.com/drive/u/2/folders/1whckadqTaQhw5sZ_dIqQZhtaxjL8Lzts" --header="Cookie: AUTH_5duq3i08p8lb90k7bjdi6mv4mkuct380_nonce=8t7cdn4f717ui" --header="Connection: keep-alive" "https://doc-14-5s-docs.googleusercontent.com/docs/securesc/ijsjbo7j24dg1fshcvqn84h4ucoebo1g/n68kl1tmithfmprktsbr8v1d47n2rs47/1566432000000/16497152722325373235/16497152722325373235/1oZ5llnA4btD9LSmnSB0GaZtTogskLwCe?e=download&h=00885983406768461781&nonce=8t7cdn4f717ui&user=16497152722325373235&hash=qrljff7cmfnp6tur08smia9l8ifbmji0" -O "ds_csv_ann.zip" -c

unzip ds_csv_ann.zip
mv ds_csv_ann/* .
rmdir ds_csv_ann
rm ds_csv_ann.zip
