import json, math, sys
from shapely.geometry import shape, box, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
def load(fn): return [shape(f['geometry']) for f in json.load(open(fn))['features']]
def proj_path(geom, lon0, lat0, k, W, H, tol):
    c=math.cos(math.radians(lat0))
    def P(x,y): return ((x-lon0)*k*c+W/2, -(y-lat0)*k+H/2)
    g=geom.simplify(tol, preserve_topology=True)
    polys=[g] if isinstance(g,Polygon) else list(getattr(g,'geoms',[]))
    out=[]
    for p in polys:
        if p.is_empty or p.area<1e-7: continue
        for ring in [p.exterior]+list(p.interiors):
            pts=[P(x,y) for x,y in ring.coords]
            out.append("M"+"L".join(f"{x:.1f},{y:.1f}" for x,y in pts)+"Z")
    return "".join(out)
W,H=1920,1080
adm3=load('adm3.geojson'); adm0=load('adm0.geojson'); ne=load('ne50land.geojson')
res={}
# local view
lon0,lat0=79.86,9.63; k=W/0.62/math.cos(math.radians(lat0))
bb=box(79.3,9.2,80.5,10.0)
land=unary_union([g.buffer(0.0003).intersection(bb) for g in adm3 if g.intersects(bb)]).buffer(-0.0003)
res['local']={'lon0':lon0,'lat0':lat0,'k':k,'path':proj_path(land,lon0,lat0,k,W,H,0.0004)}
# region view: Sri Lanka + south India
lon0,lat0=80.6,8.4; k=H/7.5
bb=box(70,0,92,16)
sl=unary_union(adm0)
reg=unary_union([g.intersection(bb) for g in ne if g.intersects(bb)]).difference(sl.buffer(0.05))
res['region']={'lon0':lon0,'lat0':lat0,'k':k,'sl':proj_path(sl,lon0,lat0,k,W,H,0.01),'india':proj_path(reg,lon0,lat0,k,W,H,0.02)}
# world for globe: raw lon/lat rings, simplified
rings=[]
for g in ne:
    g=g.simplify(0.4)
    for p in ([g] if isinstance(g,Polygon) else g.geoms):
        if p.area<0.5: continue
        rings.append([[round(x,1),round(y,1)] for x,y in p.exterior.coords])
res['world']=rings
json.dump(res,open(sys.argv[1],'w'))
print({k:len(json.dumps(v)) for k,v in res.items()})
