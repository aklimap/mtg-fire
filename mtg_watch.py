"""
mtg_watch.py — Surveillance des feux via MTG (geostationnaire, ~10 min).

Telecharge le dernier fichier feu MTG, extrait les feux categorie 2 et 3,
les convertit en lat/lon et filtre sur l'Algerie.

Cles lues depuis l'environnement : EUMETSAT_KEY, EUMETSAT_SECRET
Dependances : eumdac, xarray, netCDF4, pyproj, numpy
Lancer :  py mtg_watch.py
"""

import glob
import os
import shutil
import tempfile
import zipfile

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

import eumdac

KEY = os.environ.get("EUMETSAT_KEY", "")
SECRET = os.environ.get("EUMETSAT_SECRET", "")

COLLECTION_FEU = "EO:EUM:DAT:0682"   # Active Fire Monitoring (netCDF) - MTG
CODES_FEU = [2, 3]                    # categories fiables uniquement
AOI = {"ouest": -2.5, "sud": 34.0, "est": 8.7, "nord": 37.1}


def telecharger_dernier(dossier):
    """Telecharge le fichier feu MTG le plus recent. Renvoie le chemin du zip."""
    token = eumdac.AccessToken((KEY, SECRET))
    datastore = eumdac.DataStore(token)
    collection = datastore.get_collection(COLLECTION_FEU)
    dernier = None
    for p in collection.search():
        dernier = p
        break  # le premier est le plus recent
    if dernier is None:
        return None
    print(f"Produit le plus recent : {dernier}")
    with dernier.open() as fsrc:
        chemin = os.path.join(dossier, fsrc.name)
        with open(chemin, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
    return chemin


def extraire_nc(chemin_zip, dossier):
    if zipfile.is_zipfile(chemin_zip):
        with zipfile.ZipFile(chemin_zip) as z:
            z.extractall(dossier)
        ncs = glob.glob(os.path.join(dossier, "**", "*.nc"), recursive=True)
        return ncs[0] if ncs else None
    return chemin_zip if chemin_zip.endswith(".nc") else None


def extraire_feux_algerie(nc):
    ds = xr.open_dataset(nc, drop_variables=["fire_probability"])
    debut = ds.attrs.get("time_coverage_start")
    fin = ds.attrs.get("time_coverage_end")
    fr = ds["fire_result"].values
    x_rad = ds["x"].values
    y_rad = ds["y"].values
    masque = np.isin(fr, CODES_FEU)
    lignes, colonnes = np.where(masque)
    if len(lignes) == 0:
        return [], debut, fin
    proj = ds["mtg_geos_projection"].attrs
    h = float(proj["perspective_point_height"])
    x_m = -x_rad[colonnes] * h
    y_m = y_rad[lignes] * h
    crs_geos = CRS.from_cf({
        "grid_mapping_name": "geostationary",
        "perspective_point_height": h,
        "semi_major_axis": float(proj["semi_major_axis"]),
        "semi_minor_axis": float(proj["semi_minor_axis"]),
        "longitude_of_projection_origin": float(proj["longitude_of_projection_origin"]),
        "latitude_of_projection_origin": float(proj["latitude_of_projection_origin"]),
        "sweep_angle_axis": proj["sweep_angle_axis"],
    })
    transformer = Transformer.from_crs(crs_geos, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x_m, y_m)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    dans = (
        np.isfinite(lat) & np.isfinite(lon) &
        (lon >= AOI["ouest"]) & (lon <= AOI["est"]) &
        (lat >= AOI["sud"]) & (lat <= AOI["nord"])
    )
    feux = [
        {"lat": round(float(la), 4), "lon": round(float(lo), 4),
         "categorie": int(fr[li, co])}
        for la, lo, li, co in zip(lat[dans], lon[dans], lignes[dans], colonnes[dans])
    ]
    return feux, debut, fin


def main():
    if not KEY or not SECRET:
        print("ERREUR : EUMETSAT_KEY / EUMETSAT_SECRET absents de l'environnement.")
        return
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = telecharger_dernier(tmp)
        if not zip_path:
            print("Aucun produit telecharge.")
            return
        nc = extraire_nc(zip_path, tmp)
        if not nc:
            print("Pas de .nc dans le telechargement.")
            return
        feux, debut, fin = extraire_feux_algerie(nc)
    print(f"Periode : {debut} - {fin} UTC")
    print(f"Feux (cat. 2-3) au-dessus de l'Algerie : {len(feux)}\n")
    for f in feux:
        print(f"  lat={f['lat']}  lon={f['lon']}  categorie={f['categorie']}")
    if not feux:
        print("(Aucun feu fiable en Algerie sur ce creneau — normal si pas d'incendie.)")


if __name__ == "__main__":
    main()
