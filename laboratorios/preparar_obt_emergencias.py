#!/usr/bin/env python3
"""Construye localmente la OBT diaria del ECU 911 desde el CSV crudo unificado.

La transformación reproduce, en una versión local, la lógica Silver/Gold/OBT del
PSet 2. DuckDB procesa el archivo por bloques y puede usar disco temporal, por lo
que no intenta cargar los 2,76 GiB completos en memoria.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    laboratorios = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Genera una fila por canton y dia a partir del CSV crudo del ECU 911."
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=laboratorios
        / "semana-08-sesion-02"
        / "data"
        / "ecu911_emergencias_raw.csv",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=laboratorios
        / "semana-08-sesion-02"
        / "data"
        / "obt_emergencias_canton_dia.csv",
    )
    parser.add_argument(
        "--enlace",
        type=Path,
        default=laboratorios
        / "semana-09-sesion-01"
        / "data"
        / "obt_emergencias_canton_dia.csv",
    )
    parser.add_argument(
        "--referencias",
        type=Path,
        default=laboratorios / "datos_referencia_ecu911",
    )
    parser.add_argument("--forzar", action="store_true")
    return parser.parse_args()


def sql_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def create_hard_link(source: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists():
        link.unlink()
    os.link(source, link)
    if not os.path.samefile(source, link):
        raise RuntimeError("Las dos rutas de la OBT no apuntan al mismo archivo fisico")


def scalar(connection, query: str):
    return connection.execute(query).fetchone()[0]


def main() -> int:
    args = parse_args()
    raw = args.entrada.resolve()
    output = args.salida.resolve()
    link = args.enlace.resolve()
    holidays = (args.referencias / "feriados_ecuador.csv").resolve()
    reassigned = (args.referencias / "cantones_reasignados.csv").resolve()

    for required in (raw, holidays, reassigned):
        if not required.exists():
            raise FileNotFoundError(f"No se encontro el archivo requerido: {required}")
    if output == link:
        raise ValueError("--salida y --enlace deben ser rutas distintas")
    if output.exists() and not args.forzar:
        raise FileExistsError(
            f"La OBT ya existe: {output}. Usa --forzar para regenerarla."
        )

    try:
        import duckdb
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Falta DuckDB. Ejecuta el generador mediante el servicio datos-proyecto."
        ) from error

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)

    raw_sql = sql_path(raw)
    holidays_sql = sql_path(holidays)
    reassigned_sql = sql_path(reassigned)
    partial_sql = sql_path(partial)

    try:
        with tempfile.TemporaryDirectory(
            prefix="duckdb_ecu911_", dir=output.parent
        ) as temporary_directory:
            connection = duckdb.connect()
            try:
                threads = min(4, os.cpu_count() or 2)
                connection.execute(f"SET threads={threads}")
                connection.execute("SET memory_limit='2GB'")
                connection.execute("SET preserve_insertion_order=false")
                connection.execute(
                    f"SET temp_directory={sql_path(Path(temporary_directory))}"
                )

                print("[1/7] Leyendo y normalizando el CSV crudo...", flush=True)
                connection.execute(
                    f"""
                    CREATE TABLE limpio AS
                    SELECT
                        CASE
                            WHEN try_strptime(trim(Fecha), '%d/%m/%Y') IS NOT NULL
                                THEN CAST(try_strptime(trim(Fecha), '%d/%m/%Y') AS DATE)
                            WHEN regexp_full_match(trim(Fecha), '[0-9]+(\\.[0-9]+)?')
                                THEN DATE '1899-12-30'
                                     + CAST(round(CAST(trim(Fecha) AS DOUBLE)) AS INTEGER)
                        END AS fecha,
                        nullif(nullif(nullif(replace(trim(provincia), 'Ð', 'Ñ'), ''), 'NULL'), '0') AS provincia,
                        nullif(nullif(nullif(replace(trim(Canton), 'Ð', 'Ñ'), ''), 'NULL'), '0') AS canton,
                        nullif(nullif(nullif(replace(trim(Parroquia), 'Ð', 'Ñ'), ''), 'NULL'), '0') AS parroquia,
                        nullif(nullif(nullif(replace(trim(Servicio), 'Ð', 'Ñ'), ''), 'NULL'), '0') AS servicio,
                        nullif(nullif(nullif(replace(trim(Subtipo), 'Ð', 'Ñ'), ''), 'NULL'), '0') AS subtipo,
                        CASE
                            WHEN regexp_full_match(trim(Cod_Parroquia), '[0-9]{{5,6}}')
                                THEN lpad(trim(Cod_Parroquia), 6, '0')
                        END AS cod_parroquia,
                        _periodo AS periodo
                    FROM read_csv(
                        {raw_sql}, delim=';', header=true, all_varchar=true,
                        quote='"', escape='"', strict_mode=true
                    )
                    """
                )
                raw_rows = scalar(connection, "SELECT count(*) FROM limpio")
                invalid_dates = scalar(
                    connection, "SELECT count(*) FROM limpio WHERE fecha IS NULL"
                )
                if invalid_dates:
                    raise RuntimeError(
                        f"Hay {invalid_dates:,} filas cuya fecha no se pudo interpretar"
                    )
                print(f"      {raw_rows:,} incidentes leidos.", flush=True)

                print("[2/7] Resolviendo codigos y cambios territoriales...", flush=True)
                connection.execute(
                    """
                    CREATE TABLE codigo_canton AS
                    WITH frecuencias AS (
                        SELECT provincia, canton, left(cod_parroquia, 4) AS cod_canton,
                               count(*) AS n
                        FROM limpio
                        WHERE provincia IS NOT NULL AND canton IS NOT NULL
                          AND cod_parroquia IS NOT NULL
                        GROUP BY 1, 2, 3
                    ), ordenados AS (
                        SELECT *, row_number() OVER (
                            PARTITION BY provincia, canton
                            ORDER BY n DESC, cod_canton
                        ) AS posicion
                        FROM frecuencias
                    )
                    SELECT provincia, canton, cod_canton
                    FROM ordenados WHERE posicion = 1
                    """
                )
                connection.execute(
                    f"""
                    CREATE TABLE reasignados AS
                    SELECT * FROM read_csv({reassigned_sql}, header=true, all_varchar=true)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE eventos AS
                    SELECT
                        l.fecha,
                        l.periodo,
                        l.provincia,
                        coalesce(r.canton_vigente, l.canton) AS canton,
                        left(c.cod_canton, 2) AS cod_provincia,
                        coalesce(r.cod_canton_vigente, c.cod_canton) AS cod_canton,
                        l.servicio
                    FROM limpio l
                    LEFT JOIN codigo_canton c
                      ON l.provincia = c.provincia AND l.canton = c.canton
                    LEFT JOIN reasignados r
                      ON l.provincia = r.provincia
                     AND l.canton = r.canton_fuente
                     AND l.parroquia = r.parroquia
                    WHERE l.fecha IS NOT NULL AND c.cod_canton IS NOT NULL
                    """
                )
                valid_rows = scalar(connection, "SELECT count(*) FROM eventos")
                print(
                    f"      {valid_rows:,} incidentes con ubicacion valida; "
                    f"{raw_rows - valid_rows:,} sin canton asignable.",
                    flush=True,
                )

                print("[3/7] Creando dimensiones de canton y calendario...", flush=True)
                connection.execute(
                    """
                    CREATE TABLE dim_canton_base AS
                    SELECT cod_canton, canton, cod_provincia, provincia
                    FROM eventos
                    GROUP BY 1, 2, 3, 4
                    """
                )
                duplicated_codes = scalar(
                    connection,
                    """
                    SELECT count(*) FROM (
                        SELECT cod_canton FROM dim_canton_base
                        GROUP BY cod_canton HAVING count(*) > 1
                    )
                    """,
                )
                if duplicated_codes:
                    raise RuntimeError(
                        f"Hay {duplicated_codes} codigos de canton con mas de un nombre"
                    )
                connection.execute(
                    """
                    CREATE TABLE dim_canton AS
                    SELECT *,
                        CASE
                            WHEN cod_provincia IN ('07','08','09','12','13','23','24') THEN 'Costa'
                            WHEN cod_provincia IN ('01','02','03','04','05','06','10','11','17','18') THEN 'Sierra'
                            WHEN cod_provincia IN ('14','15','16','19','21','22') THEN 'Amazonía'
                            WHEN cod_provincia = '20' THEN 'Insular'
                            WHEN cod_provincia = '90' THEN 'No delimitada'
                        END AS region
                    FROM dim_canton_base
                    """
                )
                connection.execute(
                    f"""
                    CREATE TABLE feriados AS
                    SELECT CAST(fecha AS DATE) AS fecha, nombre_feriado
                    FROM read_csv({holidays_sql}, header=true, all_varchar=true)
                    """
                )

                print("[4/7] Agregando incidentes por canton y dia...", flush=True)
                connection.execute(
                    """
                    CREATE TABLE conteos AS
                    SELECT cod_canton, fecha,
                        count(*) AS n_emergencias,
                        count(*) FILTER (WHERE servicio = 'Seguridad Ciudadana') AS n_seguridad_ciudadana,
                        count(*) FILTER (WHERE servicio = 'Gestión Sanitaria') AS n_gestion_sanitaria,
                        count(*) FILTER (WHERE servicio = 'Tránsito y Movilidad') AS n_transito_movilidad,
                        count(*) FILTER (WHERE servicio = 'Servicios Municipales') AS n_servicios_municipales,
                        count(*) FILTER (WHERE servicio = 'Gestión de Siniestros') AS n_gestion_siniestros,
                        count(*) FILTER (WHERE servicio = 'Servicio Militar') AS n_servicio_militar,
                        count(*) FILTER (WHERE servicio = 'Gestión de Riesgos') AS n_gestion_riesgos,
                        count(*) FILTER (WHERE servicio IS NULL) AS n_sin_servicio
                    FROM eventos
                    GROUP BY 1, 2
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE hecho AS
                    WITH limites AS (
                        SELECT min(fecha) AS desde, max(fecha) AS hasta FROM conteos
                    ), spine AS (
                        SELECT c.cod_canton, CAST(d.fecha AS DATE) AS fecha
                        FROM dim_canton c
                        CROSS JOIN limites l
                        CROSS JOIN generate_series(l.desde, l.hasta, INTERVAL 1 DAY) d(fecha)
                    )
                    SELECT s.cod_canton, s.fecha,
                        coalesce(x.n_emergencias, 0) AS n_emergencias,
                        coalesce(x.n_seguridad_ciudadana, 0) AS n_seguridad_ciudadana,
                        coalesce(x.n_gestion_sanitaria, 0) AS n_gestion_sanitaria,
                        coalesce(x.n_transito_movilidad, 0) AS n_transito_movilidad,
                        coalesce(x.n_servicios_municipales, 0) AS n_servicios_municipales,
                        coalesce(x.n_gestion_siniestros, 0) AS n_gestion_siniestros,
                        coalesce(x.n_servicio_militar, 0) AS n_servicio_militar,
                        coalesce(x.n_gestion_riesgos, 0) AS n_gestion_riesgos,
                        coalesce(x.n_sin_servicio, 0) AS n_sin_servicio
                    FROM spine s
                    LEFT JOIN conteos x USING (cod_canton, fecha)
                    """
                )

                print("[5/7] Calculando rezagos, medias y objetivos...", flush=True)
                connection.execute(
                    """
                    CREATE TABLE obt AS
                    WITH base AS (
                        SELECT
                            h.cod_canton, h.fecha,
                            c.canton, c.cod_provincia, c.provincia, c.region,
                            c.cod_provincia = '90' AS es_zona_no_delimitada,
                            year(h.fecha) AS anio,
                            quarter(h.fecha) AS trimestre,
                            month(h.fecha) AS mes,
                            strftime(h.fecha, '%Y-%m') AS periodo,
                            day(h.fecha) AS dia_mes,
                            isodow(h.fecha) AS dia_semana,
                            CASE isodow(h.fecha)
                                WHEN 1 THEN 'Lunes' WHEN 2 THEN 'Martes'
                                WHEN 3 THEN 'Miércoles' WHEN 4 THEN 'Jueves'
                                WHEN 5 THEN 'Viernes' WHEN 6 THEN 'Sábado'
                                WHEN 7 THEN 'Domingo'
                            END AS nombre_dia,
                            isodow(h.fecha) >= 6 AS es_fin_de_semana,
                            f.fecha IS NOT NULL AS es_feriado,
                            f.nombre_feriado,
                            strftime(h.fecha, '%Y-%m') = '2024-01' AS es_periodo_anomalo,
                            h.n_emergencias, h.n_seguridad_ciudadana,
                            h.n_gestion_sanitaria, h.n_transito_movilidad,
                            h.n_servicios_municipales, h.n_gestion_siniestros,
                            h.n_servicio_militar, h.n_gestion_riesgos, h.n_sin_servicio,
                            f3.fecha IS NOT NULL AS es_feriado_t3,
                            f7.fecha IS NOT NULL AS es_feriado_t7
                        FROM hecho h
                        JOIN dim_canton c USING (cod_canton)
                        LEFT JOIN feriados f ON h.fecha = f.fecha
                        LEFT JOIN feriados f3 ON h.fecha + INTERVAL 3 DAY = f3.fecha
                        LEFT JOIN feriados f7 ON h.fecha + INTERVAL 7 DAY = f7.fecha
                    )
                    SELECT
                        cod_canton, fecha, canton, cod_provincia, provincia, region,
                        es_zona_no_delimitada, anio, trimestre, mes, periodo, dia_mes,
                        dia_semana, nombre_dia, es_fin_de_semana, es_feriado,
                        nombre_feriado, es_periodo_anomalo,
                        n_emergencias, n_seguridad_ciudadana, n_gestion_sanitaria,
                        n_transito_movilidad, n_servicios_municipales,
                        n_gestion_siniestros, n_servicio_militar, n_gestion_riesgos,
                        n_sin_servicio,
                        lag(n_emergencias, 1) OVER w AS lag_1d,
                        lag(n_emergencias, 7) OVER w AS lag_7d,
                        lag(n_emergencias, 14) OVER w AS lag_14d,
                        lag(n_emergencias, 28) OVER w AS lag_28d,
                        CASE WHEN count(n_emergencias) OVER w7 = 7
                             THEN avg(n_emergencias) OVER w7 END AS media_prev_7d,
                        CASE WHEN count(n_emergencias) OVER w28 = 28
                             THEN avg(n_emergencias) OVER w28 END AS media_prev_28d,
                        es_feriado_t3, es_feriado_t7,
                        lead(n_emergencias, 3) OVER w AS obj_n_emergencias_t3,
                        lead(n_emergencias, 7) OVER w AS obj_n_emergencias_t7
                    FROM base
                    WINDOW
                        w AS (PARTITION BY cod_canton ORDER BY fecha),
                        w7 AS (PARTITION BY cod_canton ORDER BY fecha ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING),
                        w28 AS (PARTITION BY cod_canton ORDER BY fecha ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING)
                    """
                )

                print("[6/7] Ejecutando controles de integridad...", flush=True)
                obt_rows = scalar(connection, "SELECT count(*) FROM obt")
                unique_keys = scalar(
                    connection,
                    "SELECT count(*) FROM (SELECT DISTINCT cod_canton, fecha FROM obt)",
                )
                emergency_sum = scalar(
                    connection, "SELECT sum(n_emergencias) FROM obt"
                )
                service_sum = scalar(
                    connection,
                    """
                    SELECT sum(n_seguridad_ciudadana + n_gestion_sanitaria
                             + n_transito_movilidad + n_servicios_municipales
                             + n_gestion_siniestros + n_servicio_militar
                             + n_gestion_riesgos + n_sin_servicio)
                    FROM obt
                    """,
                )
                cantons = scalar(connection, "SELECT count(*) FROM dim_canton")
                dates = scalar(connection, "SELECT count(DISTINCT fecha) FROM obt")
                min_date, max_date = connection.execute(
                    "SELECT min(fecha), max(fecha) FROM obt"
                ).fetchone()
                null_t3 = scalar(
                    connection,
                    "SELECT count(*) FROM obt WHERE obj_n_emergencias_t3 IS NULL",
                )
                null_t7 = scalar(
                    connection,
                    "SELECT count(*) FROM obt WHERE obj_n_emergencias_t7 IS NULL",
                )
                checks = {
                    "grano canton-dia unico": unique_keys == obt_rows,
                    "spine completo": obt_rows == cantons * dates,
                    "emergencias conservadas": emergency_sum == valid_rows,
                    "servicios suman el total": service_sum == emergency_sum,
                    "nulos t+3 esperados": null_t3 == cantons * 3,
                    "nulos t+7 esperados": null_t7 == cantons * 7,
                }
                for description, passed in checks.items():
                    print(
                        f"      {'OK' if passed else 'FALLA'}: {description}",
                        flush=True,
                    )
                if not all(checks.values()):
                    raise RuntimeError("La OBT no se escribio porque fallo un control")

                print("[7/7] Escribiendo el CSV derivado...", flush=True)
                connection.execute(
                    f"COPY (SELECT * FROM obt ORDER BY cod_canton, fecha) "
                    f"TO {partial_sql} (HEADER, DELIMITER ',')"
                )
            finally:
                connection.close()

        os.replace(partial, output)
        create_hard_link(output, link)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    print("\nOBT local terminada.", flush=True)
    print(f"Periodo: {min_date} a {max_date}", flush=True)
    print(f"Cantones: {cantons:,} | Dias: {dates:,} | Filas: {obt_rows:,}", flush=True)
    print(f"Emergencias validas: {emergency_sum:,} de {raw_rows:,}", flush=True)
    print(f"Tamano: {output.stat().st_size / (1024**2):.1f} MiB", flush=True)
    print(f"Salida: {output}", flush=True)
    print(f"Enlace: {link}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nProceso cancelado; se elimino la salida parcial.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
