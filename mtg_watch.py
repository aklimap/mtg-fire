"""
mtg_watch.py — Surveillance des feux via MTG (geostationnaire, ~10 min).

Telecharge le dernier fichier feu MTG, extrait les feux categorie 2 et 3,
les convertit en lat/lon (avec correction du signe x), filtre sur l'Algerie,
puis SUIT CHAQUE FOYER DANS LE TEMPS d'une trame a l'autre.

C'est le coeur de la validation : un geostationnaire revoit la meme scene
toutes les ~10 minutes. Un vrai feu laisse une signature qui persiste sur
plusieurs trames ; un faux positif (reflet solaire, bruit capteur, pixel
chaud isole) apparait sur une seule trame puis disparait. Un foyer vu sur
au moins deux trames distinctes est marque CONFIRME.

Resultat dans feux_mtg.json (carte web), etat du suivi dans etat_mtg.json.

Cles lues depuis l'environnement : EUMETSAT_KEY, EUMETSAT_SECRET
Dependances : eumdac, xarray, netCDF4, pyproj, numpy
Lancer :  py mtg_watch.py
"""

import glob
import json
import math
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

import eumdac

KEY = os.environ.get("EUMETSAT_KEY", "")
SECRET = os.environ.get("EUMETSAT_SECRET", "")

COLLECTION_FEU = "EO:EUM:DAT:0682"   # Active Fire Monitoring (netCDF) - MTG
CODES_FEU = [2, 3]                    # categories fiables uniquement
AOI = {"ouest": -8.7, "sud": 18.9, "est": 12.0, "nord": 37.1}   # meme zone que FIRMS (toute l Algerie)
SORTIE_JSON = "feux_mtg.json"        # fichier lu par la carte web
ETAT_JSON = "etat_mtg.json"          # etat du suivi temporel entre executions

# 24 h : aligne sur la fenetre FIRMS (DAY_RANGE = 1), sans quoi la quasi-
# totalite des detections FIRMS serait trop ancienne pour etre croisee.
FENETRE_HEURES = 24.0       # un foyer non revu depuis 24 h sort du suivi
RAYON_ASSOC_KM = 3.5        # rayon d'association (pixel FCI ~2-3 km en Algerie)
SEUIL_CONFIRMATION = 2      # trames distinctes requises pour confirmer un feu
MAX_TRAMES_GARDEES = 60     # borne la taille de l'etat

# Masque des sources thermiques permanentes, partage avec la chaine FIRMS
# (torcheres, raffineries) : sans lui, MTG suivrait une torchere pendant 24 h
# et la presenterait comme un feu confirme.
URL_SOURCES = ("https://raw.githubusercontent.com/aklimap/fire-watch/main/"
               "sources_permanentes.json")


def telecharger_dernier(dossier):
    token = eumdac.AccessToken((KEY, SECRET))
    datastore = eumdac.DataStore(token)
    collection = datastore.get_collection(COLLECTION_FEU)
    dernier = None
    for p in collection.search():
        dernier = p
        break
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
    # Correction du signe x (axe geostationnaire oriente vers l'ouest)
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


def charger_sources_permanentes():
    """Recupere le masque des sources industrielles. Renvoie [] si indisponible."""
    try:
        import urllib.request
        horodatage = int(datetime.now(timezone.utc).timestamp())
        req = urllib.request.Request(f"{URL_SOURCES}?t={horodatage}",
                                     headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        defaut = data.get("rayon_km_defaut", 3.0)
        src = [(x["lon"], x["lat"], x.get("rayon_km", defaut))
               for x in data.get("sources", [])]
        print(f"Masque des sources permanentes : {len(src)} site(s).")
        return src
    except Exception as exc:
        print(f"[!] Masque des sources permanentes indisponible : {exc}")
        return []


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance entre deux points, en kilometres."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _horodatage(slot):
    """Convertit '20261122011000' en datetime UTC."""
    return datetime.strptime(slot, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def charger_etat():
    if os.path.exists(ETAT_JSON):
        try:
            with open(ETAT_JSON, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] etat_mtg.json illisible ({e}) : on repart a zero.")
    return {"sites": []}


def sauver_etat(sites):
    with open(ETAT_JSON, "w", encoding="utf-8") as f:
        json.dump({"mis_a_jour": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "sites": sites}, f, ensure_ascii=False, indent=2)


def est_permanente(lat, lon, sources):
    return any(haversine_km(lat, lon, la, lo) <= r for lo, la, r in sources)


def suivre_dans_le_temps(feux, slot, etat, sources=()):
    """Associe les detections de la trame courante aux foyers deja suivis.

    Un foyer vu sur plusieurs trames distinctes est un vrai feu : sa signature
    thermique persiste. Un pixel vu une seule fois reste EN_ATTENTE tant
    qu'une deuxieme trame ne le confirme pas.
    """
    sites = etat.get("sites", [])
    maintenant = datetime.now(timezone.utc)
    sites = [s for s in sites
             if (maintenant - _horodatage(s["derniere"])).total_seconds()
             <= FENETRE_HEURES * 3600]

    for f in feux:
        candidat, meilleure = None, RAYON_ASSOC_KM
        for s in sites:
            d = haversine_km(f["lat"], f["lon"], s["lat"], s["lon"])
            if d <= meilleure:
                candidat, meilleure = s, d

        if candidat is None:
            sites.append({"lat": f["lat"], "lon": f["lon"],
                          "categorie_max": f["categorie"],
                          "trames": [slot], "premiere": slot, "derniere": slot})
            continue

        # On n'ajoute la trame que si elle est nouvelle : sinon deux executions
        # dans le meme creneau compteraient le meme foyer deux fois.
        if slot not in candidat["trames"]:
            candidat["trames"] = (candidat["trames"] + [slot])[-MAX_TRAMES_GARDEES:]
        candidat["derniere"] = max(candidat["derniere"], slot)
        candidat["premiere"] = min(candidat["premiere"], slot)
        candidat["categorie_max"] = max(candidat["categorie_max"], f["categorie"])
        n = len(candidat["trames"])
        candidat["lat"] = round((candidat["lat"] * (n - 1) + f["lat"]) / n, 4)
        candidat["lon"] = round((candidat["lon"] * (n - 1) + f["lon"]) / n, 4)

    maintenant = datetime.now(timezone.utc)
    for s in sites:
        s["n_trames"] = len(s["trames"])
        s["derniere_vue_min"] = int(
            (maintenant - _horodatage(s["derniere"])).total_seconds() / 60)
        s["permanente"] = est_permanente(s["lat"], s["lon"], sources)
        s["duree_min"] = int((_horodatage(s["derniere"])
                              - _horodatage(s["premiere"])).total_seconds() / 60)
        s["confirme"] = (s["n_trames"] >= SEUIL_CONFIRMATION
                         and not s["permanente"])
        if s["permanente"]:
            s["statut"] = "PERMANENT"
        else:
            s["statut"] = "CONFIRME" if s["confirme"] else "EN_ATTENTE"

    sites.sort(key=lambda s: (-s["n_trames"], -s["categorie_max"]))
    return sites


def ecrire_json(sites, debut, fin):
    """Ecrit les foyers suivis dans feux_mtg.json pour la carte web."""
    feux = [{"lat": s["lat"], "lon": s["lon"], "categorie": s["categorie_max"],
             "statut": s["statut"], "n_trames": s["n_trames"],
             "duree_min": s["duree_min"], "premiere_detection": s["premiere"],
             "derniere_detection": s["derniere"],
             "derniere_vue_min": s["derniere_vue_min"]} for s in sites]
    confirmes = sum(1 for s in sites if s["confirme"])
    data = {"source": "MTG",
            "mis_a_jour": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "periode_debut": debut, "periode_fin": fin,
            "nombre": len(feux), "nombre_confirmes": confirmes,
            "seuil_confirmation_trames": SEUIL_CONFIRMATION,
            "fenetre_suivi_heures": FENETRE_HEURES,
            "feux": feux}
    with open(SORTIE_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Ecrit dans {SORTIE_JSON} ({len(feux)} foyers, {confirmes} confirme(s))")


def main():
    if not KEY or not SECRET:
        print("ERREUR : EUMETSAT_KEY / EUMETSAT_SECRET absents de l'environnement.")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
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
    print(f"Detections (cat. 2-3) sur cette trame : {len(feux)}")

    sources = charger_sources_permanentes()
    sites = suivre_dans_le_temps(feux, debut, charger_etat(), sources)
    sauver_etat(sites)

    confirmes = [s for s in sites if s["confirme"]]
    perm = [s for s in sites if s["permanente"]]
    print(f"Foyers suivis sur {FENETRE_HEURES} h : {len(sites)} "
          f"({len(confirmes)} confirme(s), {len(perm)} source(s) permanente(s))")
    for s in sites[:20]:
        print(f"  {s['statut']:10s} lat={s['lat']:.4f} lon={s['lon']:.4f} "
              f"cat={s['categorie_max']} trames={s['n_trames']} "
              f"duree={s['duree_min']} min")
    if not sites:
        print("(Aucun foyer suivi - normal en l'absence d'incendie.)")

    ecrire_json(sites, debut, fin)


if __name__ == "__main__":
    main()
