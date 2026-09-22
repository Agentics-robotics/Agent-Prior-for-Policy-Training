import numpy as np
import cv2


def color_seed(rgb):
    x=rgb.astype(np.float32)
    return ((x[:,:,0]>x[:,:,1]*1.025)&(x[:,:,1]>x[:,:,2]*1.11)&(x[:,:,0]-x[:,:,2]>12)&(x[:,:,0]>22)&(x[:,:,0]<225)).astype(np.uint8)


def roi_default(shape):
    m=np.zeros(shape[:2],np.uint8);h,w=shape[:2];p=np.array([[.212,.058],[.605,.035],[.828,.998],[.028,.998]])*np.array([w,h]);cv2.fillPoly(m,[p.astype(np.int32)],1);return m


def repair_holes(mask,rgb):
    x=rgb.astype(np.float32);neutral=((np.max(x,axis=2)-np.min(x,axis=2))<.13*np.maximum(np.mean(x,axis=2),1))&(x[:,:,1]>35)
    ins=cv2.erode(mask,np.ones((3,3),np.uint8));n,l,stats,cent=cv2.connectedComponentsWithStats((neutral&ins.astype(bool)).astype(np.uint8));out=mask.copy()
    for j in range(1,n):
        if 140<stats[j,cv2.CC_STAT_AREA]<8000:
            part=(l==j).astype(np.uint8);ring=cv2.dilate(part,np.ones((7,7),np.uint8))-part
            if (ring*(1-mask)).sum()<4:
                out[part>0]=0
    return out


def reference_proposals(rgb,predictor):
    # Whole foreground-component boxes at the causal precontact reference.
    # Point masks may select semantic subparts of an otherwise visible bar.
    roi=roi_default(rgb.shape);seed=cv2.morphologyEx(color_seed(rgb)*roi,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));cs,_=cv2.findContours(seed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);boxes=[]
    for c in cs:
        if 450<cv2.contourArea(c)<46000:
            x,y,w,h=cv2.boundingRect(c);boxes.append([max(0,x-7),max(0,y-7),min(rgb.shape[1]-1,x+w+7),min(rgb.shape[0]-1,y+h+7)])
    predictor.set_image(rgb);out=[]
    if boxes:
        mm,ss,_=predictor.predict(box=np.array(boxes),multimask_output=False)
        for m,s in zip(mm,ss):
            one=np.asarray(m).reshape(rgb.shape[:2]).astype(np.uint8)
            if float(np.max(s))>.55 and 450<one.sum()<46000:
                out.append(repair_holes(one,rgb))
    return out


def proposals(rgb,predictor,automatic=False,roi=None):
    roi=roi_default(rgb.shape) if roi is None else roi;predictor.set_image(rgb);masks=[]
    if automatic:
        points=[]
        for y in np.linspace(25,rgb.shape[0]-25,20):
            for x in np.linspace(25,rgb.shape[1]-25,28):
                if roi[int(y),int(x)]:
                    points.append([x,y])
        for j in range(0,len(points),32):
            p=np.array(points[j:j+32],np.float32)[:,None,:];mm,ss,_=predictor.predict(point_coords=p,point_labels=np.ones(p.shape[:2],np.int32),multimask_output=True)
            for m,s in zip(mm,ss):
                for one,score in zip(m,s):
                    a=int(one.sum())
                    if score>.8 and 450<a<42000 and (one*roi).sum()>.95*a:
                        masks.append(one.astype(np.uint8))
    else:
        seed=cv2.morphologyEx(color_seed(rgb)*roi,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));count,l,stats,cent=cv2.connectedComponentsWithStats(seed);points=[]
        for j in range(1,count):
            x,y,w,h,a=stats[j]
            if not 450<a<46000 or min(w,h)<15:
                continue
            dist=cv2.distanceTransform((l==j).astype(np.uint8),cv2.DIST_L2,5)
            for k in range(3):
                yy,xx=np.unravel_index(int(dist.argmax()),dist.shape)
                if dist[yy,xx]<5:
                    break
                points.append([xx,yy]);cv2.circle(dist,(int(xx),int(yy)),int(max(25,.33*max(w,h))),0,-1)
        for j in range(0,len(points),32):
            p=np.array(points[j:j+32],np.float32)[:,None,:];mm,ss,_=predictor.predict(point_coords=p,point_labels=np.ones(p.shape[:2],np.int32),multimask_output=True)
            for m,s in zip(mm,ss):
                options=[]
                for one,score in zip(m,s):
                    a=int(one.sum())
                    if score>.65 and 450<a<42000 and (one*roi).sum()>.94*a and (one*seed).sum()>.40*a:
                        options.append((float(score),one))
                if options:
                    options.sort(key=lambda item:item[0],reverse=True);masks.append(options[0][1].astype(np.uint8))
    unique=[]
    for m in masks:
        a=int(m.sum())
        if a<450 or a>42000 or (m*roi).sum()<.93*a:
            continue
        if not any((m*k).sum()/max(1,(m|k).sum())>.67 for k in unique):
            unique.append(m)
    kept=[]
    for i,m in enumerate(unique):
        smaller=[k for j,k in enumerate(unique) if j!=i and k.sum()<.8*m.sum() and (m*k).sum()>.9*k.sum()];union=np.zeros_like(m)
        for k in smaller:
            union|=k
        if len(smaller)>=2 and (union*m).sum()>.72*m.sum():
            continue
        kept.append(repair_holes(m,rgb))
    return kept


def visible_tip(rgb,nominal):
    h,w=rgb.shape[:2];nx,ny=np.asarray(nominal,float)
    if not 10<nx<w-10 or not 25<ny<h-10:
        return None
    x0=max(0,int(nx)-55);x1=min(w,int(nx)+56);rows=[]
    for y in range(max(0,int(ny)-85),min(h,int(ny)-15)):
        line=rgb[y,x0:x1].astype(float);gray=line.mean(1);good=(line[:,2]>.9*line[:,0])&(abs(line[:,0]-line[:,1])<18)&(gray<np.median(gray)*.9)
        edges=np.diff(np.r_[False,good,False].astype(int));left=np.where(edges==1)[0];right=np.where(edges==-1)[0];options=[(abs(x0+(a+b)/2-nx),x0+(a+b)/2) for a,b in zip(left,right) if 5<=b-a<=28]
        if options:
            options.sort();rows.append([y,options[0][1]])
    if len(rows)<25:
        return None
    p=np.array(rows);coef=np.polyfit(p[:,0],p[:,1],1);p=p[abs(np.polyval(coef,p[:,0])-p[:,1])<7]
    if len(p)<22:
        return None
    coef=np.polyfit(p[:,0],p[:,1],1)
    if abs(coef[0])>.7 or np.median(abs(np.polyval(coef,p[:,0])-p[:,1]))>3:
        return None
    last=None;gap=0
    for y in range(max(0,int(ny)-30),min(h,int(ny)+45)):
        x=int(np.polyval(coef,y))
        if x<18 or x>=w-18:
            break
        c=rgb[y,x-3:x+4].astype(float).mean(0);left=rgb[y,x-18:x-11].astype(float).mean(0);right=rgb[y,x+11:x+18].astype(float).mean(0);bg=(left.mean()+right.mean())/2
        brown_neighbors=any(v[0]>1.025*v[1] and v[1]>1.11*v[2] for v in [left,right])
        # Shaft can be brighter than MDF; do not stop at the upper material
        # edge. A bright table opening is still rejected by relative intensity.
        contrast=c.mean()<bg*.92 or (brown_neighbors and c.mean()<bg*1.25 and c.mean()<160)
        ok=c[2]>.9*c[0] and abs(c[0]-c[1])<18 and contrast
        if ok:
            last=y;gap=0
        else:
            gap+=1
            if last is not None and gap>=5:
                break
    if last is None or abs(last-ny)>40:
        return None
    return np.array([float(np.polyval(coef,last)),float(last)])
