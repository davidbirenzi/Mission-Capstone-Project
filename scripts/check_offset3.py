import pystac_client, numpy as np, rasterio
from rasterio.windows import from_bounds
cat = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
bbox=[30.07,-1.97,30.09,-1.95]
for dt in ["2017-07-02","2021-06-26","2022-07-11","2023-07-21"]:
    s = cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{dt}T00:00:00Z/{dt}T23:59:59Z", max_items=5)
    items=list(s.items())
    if not items: print(dt,"none"); continue
    it=items[0]; print("\n",dt,it.id,"baseline",it.properties.get("s2:processing_baseline"))
    vals={}
    for band in ["green","red","nir","swir16"]:
        a=it.assets[band]; rb=(a.extra_fields.get("raster:bands") or [{}])[0]
        with rasterio.open(a.href) as src:
            from rasterio.warp import transform_bounds
            l,b_,r_,t = transform_bounds("EPSG:4326", src.crs, 30.075,-1.965,30.085,-1.955)
            win=from_bounds(l,b_,r_,t,transform=src.transform)
            arr=src.read(1,window=win).astype(float)
        vals[band]=(arr, rb.get("scale"), rb.get("offset"))
        print(f"  {band}: scale={rb.get('scale')} offset={rb.get('offset')} DN p50={np.percentile(arr[arr>0],50):.0f}")
    g,r,n,sw=[vals[b][0] for b in ["green","red","nir","swir16"]]
    # resize swir (20m window may differ shape) - use percentiles approach per band median ratio approx via flattened resized
    from scipy.ndimage import zoom
    sw2 = zoom(sw, (g.shape[0]/sw.shape[0], g.shape[1]/sw.shape[1]), order=0)
    for label,go,ro,no,so in [("metadata offset applied", g*0.0001+vals['green'][2], r*0.0001+vals['red'][2], n*0.0001+vals['nir'][2], sw2*0.0001+vals['swir16'][2]),
                              ("no offset (DN/10000)", g/10000, r/10000, n/10000, sw2/10000)]:
        ok=(no>0)&(ro>0)
        ndvi=np.median(((no-ro)/(no+ro))[ok])
        ok2=(go>0)&(so>0)
        mndwi=np.median(((go-so)/(go+so))[ok2])
        print(f"   {label}: median NDVI={ndvi:.3f} MNDWI={mndwi:.3f}")
